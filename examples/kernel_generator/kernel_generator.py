import asyncio
import json
import logging
import os
import random
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from prompts_router import get_optimization_prompt, get_prompt
from RAG_helper import RAGHelper
from utils import LLMClient

logger = logging.getLogger(__name__)

from flashinfer_bench import (
    Benchmark,
    BenchmarkConfig,
    BuildSpec,
    Definition,
    EvaluationStatus,
    Solution,
    SourceFile,
    SupportedLanguages,
    Trace,
    TraceSet,
    Workload,
)

class KernelGenerator:
    def __init__(
        self,
        model_name: str,
        language: str = "cuda",
        target_gpu: str = "H100",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        reasoning_effort: str = "high",  # only used for openai reasoning models
        use_ffi: bool = True,
        rag_helper: Optional[RAGHelper] = None,
    ):
        """
        Args:
            model_name: Name of the model to use (e.g., "gpt-5")
            language: Programming language for code generation (default: "triton")
            target_gpu: Target GPU architecture (e.g., "H100", "B200", "RTX4090", default: "H100")
            api_key: API key (if None, uses LLM_API_KEY environment variable)
            base_url: Base URL for the API (need to provide for non-openai api models)
            reasoning_effort: Reasoning effort for OpenAI reasoning models ("low", "medium", "high", default: "medium")
            use_ffi: Use FFI bindings when generating CUDA kernels.
            rag_helper: Optional RAGHelper instance for retrieving relevant documentation.
        """
        self.model_name = model_name
        self.language = language
        self.target_gpu = target_gpu
        self.reasoning_effort = reasoning_effort
        self.use_ffi = use_ffi
        self.rag_helper = rag_helper
        self.rag_logs: List[Dict] = []

        self.llm = LLMClient(
            model_name=model_name,
            api_key=api_key,
            base_url=base_url,
            reasoning_effort=reasoning_effort,
        )

    def _get_supported_language(self) -> SupportedLanguages:
        language_map = {
            "python": SupportedLanguages.PYTHON,
            "triton": SupportedLanguages.TRITON,
            "cuda": SupportedLanguages.CUDA,
        }
        if self.language.lower() in language_map:
            return language_map[self.language.lower()]
        else:
            return SupportedLanguages.PYTHON

    def generate(
        self,
        trace_set: TraceSet,
        definition: Definition,
        gen_rounds: int = 10,
        beam: bool = False,
        beam_width: int = 3,
    ) -> Solution:
        """
        Generate an optimized solution through iterative improvement using flashinfer-bench feedback.

        Args:
            trace_set: The TraceSet containing workloads for evaluation
            definition: The workload definition to implement kernel for
            gen_rounds: Number of generation rounds to run (or search depth if beam=True)
            beam: beam search flag, default to False as it's more expensive to run
            beam_width: Number of candidates to maintain in beam search (default: 3)

        Returns:
            Solution: a solution dataclass containing the optimized kernel code
        """
        workloads = trace_set.workloads.get(definition.name, [])
        if not workloads:
            raise ValueError(
                f"No workloads found for definition '{definition.name}' in the provided TraceSet"
            )

        selected_workload = random.choice(workloads)

        logger.info(
            "Starting generation for '%s' | model=%s lang=%s gpu=%s rounds=%d beam=%s width=%d",
            definition.name, self.model_name, self.language, self.target_gpu,
            gen_rounds, beam, beam_width,
        )
        logger.info("Selected workload %s for optimization feedback", selected_workload.workload.uuid)

        # Reset RAG logs for this generation run
        self.rag_logs = []

        if beam:
            return self._beam_search_generate(
                trace_set, definition, selected_workload, gen_rounds, beam_width
            )
        else:
            return asyncio.run(
                self._sequential_generate_async(
                    trace_set, definition, selected_workload, gen_rounds
                )
            )

    async def _sequential_generate_async(
        self, trace_set: TraceSet, definition: Definition, selected_workload, gen_rounds: int
    ) -> Solution:
        # Retrieve RAG data for the initial prompt
        initial_rag_data = ""
        if self.rag_helper:
            logger.info("[Sequential] Retrieving RAG data for initial prompt...")
            t0 = time.perf_counter()
            initial_rag_data = await self.rag_helper.retrieve(
                definition, trace=None, current_code=None
            )
            logger.info(
                "[Sequential] RAG retrieval complete (%.2fs, %d chars)",
                time.perf_counter() - t0, len(initial_rag_data),
            )
            self.rag_logs.append({
                "mode": "sequential",
                "round": 0,
                "definition": definition.name,
                "rag_data": initial_rag_data,
            })

        prompt = get_prompt(
            self.language, definition, self.target_gpu, self.use_ffi, rag_data=initial_rag_data
        )
        logger.info("[Sequential] Generating initial code (prompt length: %d chars)...", len(prompt))
        t0 = time.perf_counter()
        code_result = await self._generate_code_from_prompt(prompt)
        logger.info("[Sequential] Initial code generated (%.2fs)", time.perf_counter() - t0)
        current_code = code_result["cleaned"]
        current_raw_code = code_result["raw"]

        passing_solutions: List[Tuple[Solution, Trace]] = []
        last_solution = None
        last_trace = None

        for round_num in range(1, gen_rounds + 1):
            logger.info("[Sequential] Round %d/%d", round_num, gen_rounds)

            solution = self._create_solution_from_code(current_code, definition, round_num)
            last_solution = solution

            logger.debug("[Sequential] Evaluating solution for round %d...", round_num)
            traces = self._evaluate_solutions(trace_set, definition, [solution], selected_workload)
            trace = traces[0] if traces else None
            if trace:
                last_trace = trace
                evaluation = trace.evaluation
                logger.info("[Sequential] Round %d evaluation: %s", round_num, evaluation.status.value)

                if evaluation.status == EvaluationStatus.PASSED:
                    speedup = evaluation.performance.speedup_factor
                    logger.info("[Sequential] Round %d PASSED — speedup: %.2fx", round_num, speedup)
                    passing_solutions.append((solution, trace))
                else:
                    logger.warning("[Sequential] Round %d FAILED — %s", round_num, evaluation.status.value)
                    if evaluation.log:
                        logger.debug("[Sequential] Error details:\n%s", evaluation.log)

            if round_num < gen_rounds:
                best_trace = self._get_best_trace(passing_solutions)
                opt_trace = best_trace if best_trace else last_trace

                if opt_trace:
                    optimization_prompt = get_optimization_prompt(
                        self.language,
                        definition,
                        opt_trace,
                        current_raw_code,
                        self.target_gpu,
                        self.use_ffi,
                        rag_data=initial_rag_data,
                    )
                else:
                    optimization_prompt = get_prompt(
                        self.language, definition, self.target_gpu, self.use_ffi,
                        rag_data=initial_rag_data,
                    )

                logger.info(
                    "[Sequential] Calling LLM for round %d (prompt length: %d chars)...",
                    round_num + 1, len(optimization_prompt),
                )
                t0 = time.perf_counter()
                code_result = await self._generate_code_from_prompt(optimization_prompt)
                elapsed = time.perf_counter() - t0
                logger.info("[Sequential] LLM response received for round %d (%.2fs)", round_num + 1, elapsed)
                current_code = code_result["cleaned"]
                current_raw_code = code_result["raw"]

        return self._select_best_solution(passing_solutions, last_solution)

    def _beam_search_generate(
        self,
        trace_set: TraceSet,
        definition: Definition,
        selected_workload,
        depth: int,
        beam_width: int,
    ) -> Solution:
        logger.info("Starting beam search with width=%d, depth=%d", beam_width, depth)
        return asyncio.run(
            self._beam_search_generate_async(
                trace_set, definition, selected_workload, depth, beam_width
            )
        )

    async def _beam_search_generate_async(
        self,
        trace_set: TraceSet,
        definition: Definition,
        selected_workload,
        depth: int,
        beam_width: int,
    ) -> Solution:
        passing_solutions: List[Tuple[Solution, Trace]] = []

        # Retrieve RAG data for the initial prompt
        initial_rag_data = ""
        if self.rag_helper:
            logger.info("[Beam] Retrieving RAG data for initial prompt...")
            t0 = time.perf_counter()
            initial_rag_data = await self.rag_helper.retrieve(
                definition, trace=None, current_code=None
            )
            logger.info(
                "[Beam] RAG retrieval complete (%.2fs, %d chars)",
                time.perf_counter() - t0, len(initial_rag_data),
            )
            self.rag_logs.append({
                "mode": "beam",
                "level": 0,
                "candidate": "all",
                "definition": definition.name,
                "rag_data": initial_rag_data,
            })

        prompt = get_prompt(
            self.language, definition, self.target_gpu, self.use_ffi, rag_data=initial_rag_data
        )

        logger.info("[Beam] Level 0: Generating %d initial candidates (prompt length: %d chars)...", beam_width, len(prompt))
        t0 = time.perf_counter()
        code_results = await asyncio.gather(
            *[self._generate_code_from_prompt(prompt) for _ in range(beam_width)]
        )
        logger.info("[Beam] Level 0: %d candidates generated (%.2fs)", beam_width, time.perf_counter() - t0)

        initial_candidates = [
            {"code": code_result["cleaned"], "raw_code": code_result["raw"], "round_num": 0}
            for code_result in code_results
        ]

        solutions = [
            self._create_solution_from_code(candidate["code"], definition, 0, candidate_idx=i)
            for i, candidate in enumerate(initial_candidates)
        ]

        logger.info("[Beam] Evaluating %d initial candidates...", len(solutions))
        traces = self._evaluate_solutions(trace_set, definition, solutions, selected_workload)

        beam = []
        for i, (candidate, solution, trace) in enumerate(
            zip(initial_candidates, solutions, traces)
        ):
            if trace:
                evaluation = trace.evaluation
                speedup = (
                    evaluation.performance.speedup_factor
                    if evaluation.status == EvaluationStatus.PASSED
                    else 0.0
                )
                logger.info("[Beam] Level 0 candidate %d: %s, speedup=%.2fx", i + 1, evaluation.status.value, speedup)

                if evaluation.status == EvaluationStatus.PASSED:
                    passing_solutions.append((solution, trace))

                beam.append(
                    {
                        "solution": solution,
                        "trace": trace,
                        "code": candidate["code"],
                        "raw_code": candidate["raw_code"],
                        "speedup": speedup,
                        "round_num": 0,
                    }
                )

        beam.sort(key=lambda x: x["speedup"], reverse=True)
        beam = beam[:beam_width]
        last_solution = beam[0]["solution"] if beam else None

        for level in range(1, depth + 1):
            logger.info("[Beam] Level %d/%d: Expanding %d candidates...", level, depth, len(beam))

            prompts = [
                get_optimization_prompt(
                    self.language,
                    definition,
                    beam_item["trace"],
                    beam_item["raw_code"],
                    self.target_gpu,
                    self.use_ffi,
                    rag_data=initial_rag_data,
                )
                for beam_item in beam
            ]

            logger.info(
                "[Beam] Level %d: Calling LLM for %d candidates (prompt lengths: %s)...",
                level, len(prompts), [len(p) for p in prompts],
            )
            t0 = time.perf_counter()
            code_results = await asyncio.gather(
                *[self._generate_code_from_prompt(prompt) for prompt in prompts]
            )
            logger.info("[Beam] Level %d: LLM responses received (%.2fs)", level, time.perf_counter() - t0)

            solutions = [
                self._create_solution_from_code(
                    code_result["cleaned"], definition, level, candidate_idx=i
                )
                for i, code_result in enumerate(code_results)
            ]

            logger.info("[Beam] Level %d: Evaluating %d expanded candidates...", level, len(solutions))
            traces = self._evaluate_solutions(trace_set, definition, solutions, selected_workload)

            new_candidates = []
            for beam_idx, (code_result, solution, trace) in enumerate(
                zip(code_results, solutions, traces)
            ):
                if trace:
                    evaluation = trace.evaluation
                    speedup = (
                        evaluation.performance.speedup_factor
                        if evaluation.status == EvaluationStatus.PASSED
                        else 0.0
                    )
                    logger.info(
                        "[Beam] Level %d candidate %d: %s, speedup=%.2fx",
                        level, beam_idx + 1, evaluation.status.value, speedup,
                    )

                    if evaluation.status == EvaluationStatus.PASSED:
                        passing_solutions.append((solution, trace))

                    new_candidates.append(
                        {
                            "solution": solution,
                            "trace": trace,
                            "code": code_result["cleaned"],
                            "raw_code": code_result["raw"],
                            "speedup": speedup,
                            "round_num": level,
                        }
                    )

            if new_candidates:
                new_candidates.sort(key=lambda x: x["speedup"], reverse=True)
                beam = new_candidates[:beam_width]
                last_solution = beam[0]["solution"]
                logger.info("[Beam] Level %d complete. Top speedup: %.2fx", level, beam[0]["speedup"])
            else:
                logger.warning("[Beam] No valid candidates at level %d, stopping beam search", level)
                break

        logger.info("[Beam] Search complete. Found %d passing solutions.", len(passing_solutions))
        return self._select_best_solution(passing_solutions, last_solution)

    def _evaluate_solutions(
        self,
        trace_set: TraceSet,
        definition: Definition,
        solutions: List[Solution],
        selected_workload,
    ) -> List[Optional[Trace]]:
        if not solutions:
            return []

        temp_trace_set = TraceSet(
            root=trace_set.root,
            definitions={definition.name: definition},
            solutions={definition.name: solutions},
            workloads={definition.name: [selected_workload]},
            traces={definition.name: []},
        )

        benchmark = Benchmark(temp_trace_set, BenchmarkConfig())
        result_trace_set = benchmark.run_all()

        traces = result_trace_set.traces.get(definition.name, [])

        trace_map = {trace.solution: trace for trace in traces}
        return [trace_map.get(sol.name) for sol in solutions]

    def _get_best_trace(self, passing_solutions: List[Tuple[Solution, Trace]]) -> Optional[Trace]:
        if not passing_solutions:
            return None

        best_solution_trace = max(
            passing_solutions, key=lambda st: st[1].evaluation.performance.speedup_factor
        )
        return best_solution_trace[1]

    def _select_best_solution(
        self, passing_solutions: List[Tuple[Solution, Trace]], fallback_solution: Optional[Solution]
    ) -> Solution:
        if passing_solutions:
            best_solution_trace = max(
                passing_solutions, key=lambda st: st[1].evaluation.performance.speedup_factor
            )
            best_solution = best_solution_trace[0]
            best_speedup = best_solution_trace[1].evaluation.performance.speedup_factor
            logger.info("Returning best solution with speedup: %.2fx", best_speedup)
            return best_solution
        elif fallback_solution:
            logger.warning("No passing solutions found, returning last generated solution")
            return fallback_solution
        else:
            raise ValueError("No solutions generated")

    def save_rag_log(self, output_dir: str, definition: Definition) -> None:
        """Save the definition and RAG data collected during generation to a file.

        Args:
            output_dir: Directory where the solution is saved (e.g. solutions/op_type/def_name/).
            definition: The Definition object used for generation.
        """
        from utils import format_definition

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        log_path = output_path / "rag_log.json"

        log_data = {
            "definition": {
                "name": definition.name,
                "op_type": definition.op_type,
                "description": format_definition(definition),
            },
            "model": self.model_name,
            "language": self.language,
            "target_gpu": self.target_gpu,
            "rag_rounds": self.rag_logs,
        }

        with open(log_path, "w") as f:
            json.dump(log_data, f, indent=2)

        logger.info("RAG log saved to %s", log_path)

    def _parse_xml_files(self, code: str) -> Dict[str, str]:
        files = {}

        patterns = {
            "kernel.h": r'<header_file name="kernel\.h">(.*?)</header_file>',
            "kernel.cu": r'<cuda_file name="kernel\.cu">(.*?)</cuda_file>',
            "main.cpp": r'<cpp_file name="main\.cpp">(.*?)</cpp_file>',
        }

        for filename, pattern in patterns.items():
            match = re.search(pattern, code, re.DOTALL)
            if match:
                content = match.group(1).strip()
                files[filename] = content
            else:
                logger.warning("Could not find %s in generated code", filename)

        return files

    def _clean_generated_code(self, code: str) -> str:
        if self.language.lower() == "cuda":
            return self._parse_xml_files(code)

        if "```" in code:
            if code.startswith("```"):
                lines = code.split("\n")
                if lines[0].startswith("```"):
                    lines = lines[1:]
                code = "\n".join(lines)

            if code.endswith("```"):
                lines = code.split("\n")
                if lines and lines[-1].strip() == "```":
                    lines = lines[:-1]
                code = "\n".join(lines)

            code = code.replace("```", "")

        hex_float_pattern = r"0x[0-9a-fA-F]*\.[0-9a-fA-F]*p[-+]?\d+"
        hex_floats = re.findall(hex_float_pattern, code)

        for hex_float in hex_floats:
            try:
                if hex_float == "0x1.62e42fefa39efp-1":
                    decimal_val = "0.6931471805599453"
                elif hex_float == "0x1.71547652b82fep0":
                    decimal_val = "2.718281828459045"
                elif hex_float == "0x1.921fb54442d18p1":
                    decimal_val = "3.141592653589793"
                else:
                    decimal_val = "1.0"

                code = code.replace(hex_float, decimal_val)
            except Exception as e:
                logger.warning("Could not convert hex float %s: %s", hex_float, e)
                code = code.replace(hex_float, "1.0")

        return code

    async def _generate_code_from_prompt(self, prompt: str):
        """Generate code from prompt using async API"""
        try:
            generated_code = await self.llm.complete(prompt)
            cleaned_code = self._clean_generated_code(generated_code)
            return {"raw": generated_code, "cleaned": cleaned_code}
        except Exception as e:
            logger.error("LLM code generation failed: %s", e)
            raise

    def _create_solution_from_code(
        self, code, definition: Definition, round_num: int, candidate_idx: int = 0
    ) -> Solution:
        if self.llm.is_reasoning_model:
            solution_name = f"{self.model_name}_{definition.name}_{self.language}_optimized_r{round_num}_c{candidate_idx}_{self.reasoning_effort}"
            solution_description = f"{self.model_name} optimized kernel for {definition.name} (round {round_num}, candidate {candidate_idx}, reasoning effort: {self.reasoning_effort})"
        else:
            solution_name = f"{self.model_name}_{definition.name}_{self.language}_optimized_r{round_num}_c{candidate_idx}"
            solution_description = f"{self.model_name} optimized kernel for {definition.name} (round {round_num}, candidate {candidate_idx})"

        if self.language.lower() == "cuda" and isinstance(code, dict):
            sources = []
            for filename, content in code.items():
                sources.append(SourceFile(path=filename, content=content))

            entry_point = "main.cpp::run"
        else:
            if isinstance(code, dict):
                code = next(iter(code.values()))

            sources = [SourceFile(path="main.py", content=code)]
            entry_point = "main.py::run"

        solution = Solution(
            name=solution_name,
            definition=definition.name,
            author=self.model_name,
            spec=BuildSpec(
                language=self._get_supported_language(),
                target_hardware=[self.target_gpu],
                entry_point=entry_point,
            ),
            sources=sources,
            description=solution_description,
        )
        return solution

"""
Example script to generate optimized solutions with KernelGenerator module
"""

import json
import logging
import os
from pathlib import Path

from dotenv import load_dotenv
from kernel_generator import KernelGenerator
from RAG_helper import RAGHelper
from utils import LLMClient

from flashinfer_bench import TraceSet
from flashinfer_bench.data import save_json_file

load_dotenv()

# ---------------------------------------------------------------------------
# Logging configuration
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)-30s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def main():
    """
    Generate optimized solutions for all definitions in the trace_set.
    """
    # TODO: select model, language, target gpu, definition
    model_name = "gemini-3-flash-preview"  # Choose author-model
    language = "cuda"  # Target solution language
    target_gpu = "H100"  # Choose solution target GPU
    target_definition_name = ""  # Leave empty to generate solutions for all definitions

    # TODO: adjust local path to trace_set
    trace_set_path = "/home/zhouyayue/flashinfer-bench/flashinfer_trace_test"

    logger.info("Loading TraceSet from: %s", trace_set_path)
    trace_set = TraceSet.from_path(trace_set_path)

    all_definitions = list(trace_set.definitions.keys())

    if not all_definitions:
        logger.error("No definitions found in trace_set at '%s'.", trace_set_path)
        logger.error("Please ensure `trace_set_path` points to a valid flashinfer-trace directory.")
        return

    if target_definition_name:
        if target_definition_name in all_definitions:
            all_definitions = [target_definition_name]
            logger.info("Generating solution for '%s'", target_definition_name)
        else:
            logger.error("Definition '%s' not found in trace_set", target_definition_name)
            return

    logger.info("Found %d definitions to generate solutions", len(all_definitions))

    api_key = os.getenv("LLM_API_KEY")
    base_url = os.getenv("BASE_URL")
    if not api_key:
        logger.error(
            "Please set LLM_API_KEY environment variable or modify this script to pass api_key explicitly"
        )
        return

    # Create a shared LLM client for both generation and RAG query rewriting
    llm_client = LLMClient(
        model_name=model_name,
        api_key=api_key,
        base_url=base_url,
        reasoning_effort="high",
    )

    rag_helper = RAGHelper(llm=llm_client)

    generator = KernelGenerator(
        model_name=model_name,
        language=language,
        target_gpu=target_gpu,
        api_key=api_key,
        base_url=base_url,
        reasoning_effort="high",
        use_ffi=False,
        rag_helper=rag_helper,
    )

    total_definitions = len(all_definitions)
    successful_generations = 0
    failed_generations = 0

    logger.info("%s", "=" * 60)
    logger.info("Generating solutions for %d definitions...", total_definitions)
    logger.info("%s", "=" * 60)

    for idx, definition_name in enumerate(all_definitions, 1):
        definition = trace_set.definitions[definition_name]

        logger.info("[%d/%d] Processing definition: %s (type: %s)", idx, total_definitions, definition_name, definition.op_type)

        workloads = trace_set.workloads.get(definition_name, [])
        if not workloads:
            logger.warning("No workloads found for definition '%s' — SKIPPING", definition_name)
            failed_generations += 1
            continue

        logger.info("Found %d workloads for '%s'", len(workloads), definition_name)

        solution = None
        max_attempts = 2

        for attempt in range(1, max_attempts + 1):
            try:
                logger.info("Attempt %d/%d for '%s'", attempt, max_attempts, definition_name)

                solution = generator.generate(
                    trace_set=trace_set,
                    definition=definition,
                    gen_rounds=3,  # For our baseline, we used 10 rounds
                    # TODO: uncomment bellow to use beam search
                    # beam=True,
                    # beam_width=3,
                )

                logger.info("Successfully generated solution for '%s'", definition_name)
                break

            except Exception as e:
                logger.error("Attempt %d failed for '%s': %s", attempt, definition_name, e)
                if attempt < max_attempts:
                    logger.info("Retrying... (%d/%d)", attempt + 1, max_attempts)
                else:
                    logger.error("All attempts failed for '%s' — SKIPPING", definition_name)
                    failed_generations += 1
                    break

        if solution:
            try:
                # Create directory structure: solutions/definition-type/definition-name/
                solutions_dir = (
                    Path(trace_set_path) / "solutions" / definition.op_type / definition_name
                )
                solutions_dir.mkdir(parents=True, exist_ok=True)

                # Create filename using solution name
                solution_filename = f"{solution.name}.json"
                solution_path = solutions_dir / solution_filename

                save_json_file(solution, solution_path)

                logger.info("Solution saved to: %s", solution_path)

                # Save RAG log (definition + RAG data) alongside the solution
                generator.save_rag_log(str(solutions_dir), definition)

                successful_generations += 1

            except Exception as e:
                logger.error("Failed to save solution for '%s': %s", definition_name, e)
                failed_generations += 1

    logger.info("%s", "=" * 60)
    logger.info("GENERATION COMPLETE")
    logger.info("%s", "=" * 60)
    logger.info("Total definitions processed: %d", total_definitions)
    logger.info("Successful generations: %d", successful_generations)
    logger.info("Failed generations: %d", failed_generations)
    success_rate = (successful_generations / total_definitions * 100) if total_definitions else 0.0
    logger.info("Success rate: %.1f%%", success_rate)


if __name__ == "__main__":
    main()

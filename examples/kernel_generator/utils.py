import os
from typing import Dict, Optional

import openai
from flashinfer_bench import FFI_PROMPT_SIMPLE, Definition, EvaluationStatus, Trace


class LLMClient:
    """Shared async LLM client that handles API key resolution, client
    construction, and model-specific response dispatch (reasoning API
    for gpt-5/o3 models vs standard chat completions for everything else).

    Both ``KernelGenerator`` and ``RAGHelper`` delegate LLM calls here.
    """

    def __init__(
        self,
        model_name: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        reasoning_effort: str = "high",
    ):
        """
        Args:
            model_name: Model identifier (e.g. "gpt-5-2025-08-07", "o3", "gpt-4o-mini").
            api_key: OpenAI-compatible API key. Falls back to ``LLM_API_KEY`` env var.
            base_url: Optional base URL for non-OpenAI endpoints.
            reasoning_effort: Effort level for reasoning-capable models ("low"/"medium"/"high").
        """
        self.model_name = model_name
        self.reasoning_effort = reasoning_effort

        if api_key is None:
            api_key = os.getenv("LLM_API_KEY")
            if api_key is None:
                raise ValueError(
                    "API key must be provided or set in LLM_API_KEY environment variable"
                )

        client_kwargs: Dict = {"api_key": api_key}
        if base_url is not None:
            client_kwargs["base_url"] = base_url

        self.client = openai.AsyncOpenAI(**client_kwargs)

    @property
    def is_reasoning_model(self) -> bool:
        """Return True for models that support the reasoning/responses API."""
        return self.model_name.startswith("gpt-5") or self.model_name.startswith("o3")

    async def complete(self, prompt: str, temperature: Optional[float] = None) -> str:
        """Send a prompt and return the raw response text.

        Automatically dispatches to the reasoning API for gpt-5/o3 models
        and to the standard chat completions API for all others.

        Args:
            prompt: The user prompt to send.
            temperature: Sampling temperature (ignored for reasoning models).

        Returns:
            The assistant's response text, stripped of leading/trailing whitespace.
        """
        if self.is_reasoning_model:
            response = await self.client.responses.create(
                model=self.model_name,
                input=prompt,
                reasoning={"effort": self.reasoning_effort},
            )
            return response.output_text.strip()
        else:
            kwargs: Dict = {
                "model": self.model_name,
                "messages": [{"role": "user", "content": prompt}],
            }
            if temperature is not None:
                kwargs["temperature"] = temperature
            response = await self.client.chat.completions.create(**kwargs)
            return response.choices[0].message.content.strip()


def format_definition(definition: Definition) -> str:
    axes_str = "\nAxes:\n"
    for name, axis in definition.axes.items():
        if hasattr(axis, "value"):
            axes_str += f"  {name}: constant = {axis.value}"
        else:
            axes_str += f"  {name}: variable"
        if axis.description:
            axes_str += f" ({axis.description})"
        axes_str += "\n"

    # Format inputs
    inputs_str = "\nInputs:\n"
    for name, spec in definition.inputs.items():
        shape_str = "scalar" if spec.shape is None else f"[{', '.join(spec.shape)}]"
        inputs_str += f"  {name}: {shape_str} ({spec.dtype})"
        if spec.description:
            inputs_str += f" - {spec.description}"
        inputs_str += "\n"

    outputs_str = "\nOutputs:\n"
    for name, spec in definition.outputs.items():
        shape_str = "scalar" if spec.shape is None else f"[{', '.join(spec.shape)}]"
        outputs_str += f"  {name}: {shape_str} ({spec.dtype})"
        if spec.description:
            outputs_str += f" - {spec.description}"
        outputs_str += "\n"

    constraints_str = ""
    if definition.constraints:
        constraints_str = "\nConstraints:\n"
        for constraint in definition.constraints:
            constraints_str += f"  - {constraint}\n"

    return f"""Name: {definition.name}
Type: {definition.op_type}
{axes_str}{inputs_str}{outputs_str}{constraints_str}

Reference Implementation:
{definition.reference}"""


def format_trace_logs(trace: Trace) -> str:
    if trace.is_workload_trace() or not trace.evaluation:
        return "No evaluation logs available (workload-only trace)"

    eval_info = f"Status: {trace.evaluation.status.value}\n"
    eval_info += f"Timestamp: {trace.evaluation.timestamp}\n"

    if trace.evaluation.log:
        eval_info += f"\nExecution Log:\n{trace.evaluation.log}\n"

    if trace.evaluation.correctness:
        eval_info += f"Max relative error: {trace.evaluation.correctness.max_relative_error}\n"
        eval_info += f"Max absolute error: {trace.evaluation.correctness.max_absolute_error}\n"

    if trace.evaluation.performance:
        eval_info += f"Latency: {trace.evaluation.performance.latency_ms}ms\n"
        eval_info += f"Reference latency: {trace.evaluation.performance.reference_latency_ms}ms\n"
        eval_info += f"Speedup factor: {trace.evaluation.performance.speedup_factor}x\n"

    return eval_info

"""
This file contains the prompts for baseline agent generation.
"""

from flashinfer_bench import FFI_PROMPT_SIMPLE, Definition, EvaluationStatus, Trace
from utils import format_definition, format_trace_logs

TRITON_PROMPT = """Generate a Triton kernel optimized for {target_gpu} GPU for

{definition}

Triton Version: 3.3.1

Requirements:
- Write clean, efficient Triton code optimized for {target_gpu} architecture
- Use modern Triton syntax with proper grid computation and language features
- Include necessary imports (torch, triton, triton.language as tl)
- Implement the exact functionality described in the specification
- The reference code provides the mathematical specification but is unoptimized - your Triton implementation should match its computational accuracy while delivering high performance
- Use the definition's tensor shapes, dtypes, and axes information to guide memory access patterns and optimization strategies
- Optimize for {target_gpu} GPU characteristics (memory hierarchy, compute units, etc.)

The wrapper function MUST handle complete device management:
- Move CPU tensors to GPU if needed (use .cuda() when torch.cuda.is_available())
- Raise clear errors if CUDA is not available for GPU tensors
- Call the triton kernel with GPU tensors
- Move results back to original device of input tensors
- Handle both args and kwargs properly
- Preserve original tensor devices and restore them for outputs

IMPORTANT: Use only valid Python/Triton syntax:
- NO hexadecimal float literals (0x1.234p5) - use decimal equivalents
- NO C/CUDA specific syntax - this is Python/Triton code
- All code must be valid Python that passes ast.parse()

{rag_data}

- Expose a "run" entry point function that can be called to execute the kernel
- Return only the code, no explanations or markdown formatting

Generate complete, runnable code only - no framework will add device handling wrapper code.

Generate the implementation:"""

TRITON_OPTIMIZATION_PROMPT = """You are optimizing a Triton kernel for {target_gpu} GPU.

Original Specification:
{definition}

Current Implementation Status:
{trace_logs}

Current Implementation:
{current_code}

{error_strategy}

Requirements for the optimized implementation:
- Write clean, efficient Triton code optimized for {target_gpu} architecture
- Use modern Triton syntax with proper grid computation and language features
- Include necessary imports (torch, triton, triton.language as tl)
- Fix all identified issues from the feedback
- Maintain or improve computational accuracy
- Preserve the same function signature and device handling as specified

The wrapper function MUST handle complete device management:
- Move CPU tensors to GPU if needed (use .cuda() when torch.cuda.is_available())
- Raise clear errors if CUDA is not available for GPU tensors
- Call the triton kernel with GPU tensors
- Move results back to original device of input tensors
- Handle both args and kwargs properly
- Preserve original tensor devices and restore them for outputs

IMPORTANT: Use only valid Python/Triton syntax:
- NO hexadecimal float literals (0x1.234p5) - use decimal equivalents
- NO C/CUDA specific syntax - this is Python/Triton code
- All code must be valid Python that passes ast.parse()

{rag_data}

- Expose a "run" entry point function that can be called to execute the kernel
- Return only the improved code, no explanations or markdown formatting

Generate the corrected and optimized implementation:"""

PYTHON_PROMPT = """You are a code generator. Generate a Python implementation optimized for {target_gpu} GPU for the following specification.

Specification:
{definition}

Requirements:
- Write clean, efficient Python code optimized for {target_gpu} architecture
- Use PyTorch operations when appropriate, optimized for {target_gpu}
- Include necessary imports
- Implement the exact functionality described in the specification

{rag_data}

- Expose a "run" entry point function that can be called to execute the implementation
- Return only the code, no explanations or markdown formatting

Generate the implementation:"""

CUDA_PROMPT = """You are a code generator. Generate a CUDA kernel implementation optimized for {target_gpu} GPU for the following specification.

Specification:
{definition}

Requirements:
- Write clean, efficient CUDA C++ code optimized for {target_gpu} architecture
- Use proper CUDA syntax and memory management optimized for {target_gpu}
- Implement the exact functionality described in the specification
- The reference code provides the mathematical specification but is unoptimized - your CUDA implementation should match its computational accuracy while delivering high performance
- Use the definition's tensor shapes, dtypes, and axes information to guide memory access patterns and optimization strategies
- Optimize for {target_gpu} GPU characteristics (memory hierarchy, compute units, etc.)
- For fixed axis values, optimize specifically for those constants rather than general cases

IMPORTANT: Generate code in XML format with exactly 3 files with these strict names:

<header_file name="kernel.h">
- All CUDA kernel function declarations
- Host function declarations
- Any necessary struct/type definitions
- Include guards and necessary headers
</header_file>

<cuda_file name="kernel.cu">
- All __global__ kernel implementations
- All __device__ helper functions
- CUDA-specific optimizations and memory patterns
- Proper error checking and memory management
</cuda_file>

<cpp_file name="main.cpp">
- Host function that launches kernels
- Memory allocation and data transfer management
- Device management and error handling
- Entry point function named "run" that can be called to execute the implementation
- Handle both args and kwargs properly
- Move CPU data to GPU, execute kernels, and return results to CPU
</cpp_file>

Code Generation Guidelines:
- Use modern CUDA features appropriate for {target_gpu}
- Optimize memory coalescing and reduce bank conflicts
- Utilize shared memory effectively for data reuse
- Consider occupancy and register usage
- Implement proper error checking with cudaGetLastError()
- Use appropriate grid and block dimensions for the problem size
- Leverage constant memory for frequently accessed read-only data
- Ensure proper CUDA stream synchronization and error handling

{rag_data}

Generate the implementation:"""

CUDA_OPTIMIZATION_PROMPT = """You are optimizing a CUDA kernel for {target_gpu} GPU.

Original Specification:
{definition}

Current Implementation Status:
{trace_logs}

Current Implementation:
{current_code}

{error_strategy}

Requirements for the optimized implementation:
- Write clean, efficient CUDA C++ code optimized for {target_gpu} architecture
- Use proper CUDA syntax and modern features appropriate for {target_gpu}

{rag_data}

Generate the corrected and optimized implementation:"""

TORCH_BINDINGS_PROMPT = """
Use TORCH for your generated kernel host function and bindings

Requirements:
- Include all necessary headers (torch/extension.h, kernel.h, etc.)
- Implement the "run" function that:
  * Takes torch::Tensor arguments
  * Validates tensor properties (device, dtype, shape)
  * Extracts raw pointers using .data_ptr<T>()
  * Calls the CUDA kernel with appropriate launch configuration
  * Returns results as torch::Tensor
- Use PYBIND11_MODULE to bind the "run" function:
  * PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  *   m.def("run", &run, "Kernel execution function");
  * }
- Handle both positional args and kwargs properly
- Include proper error messages for invalid inputs

- Use torch::Tensor for all tensor arguments
- Use .device().is_cuda() to check if tensors are on GPU
- Use .dtype() to validate tensor data types
- Use .sizes() or .size(dim) to get tensor dimensions
- Use .data_ptr<float>() or .data_ptr<T>() to get raw pointers
- Call cudaDeviceSynchronize() or cudaGetLastError() for error checking
- Return torch::Tensor from the run function
- Handle exceptions gracefully with proper error messages"""

# ---------------------------------------------------------------------------
# Error-type-specific optimization strategies
# ---------------------------------------------------------------------------

STRATEGY_COMPILE_ERROR = """Optimization Strategy — COMPILE ERROR FIX:
The current implementation FAILED TO COMPILE. This is the highest priority to fix.
- Carefully read the compilation error messages in the logs above
- Fix all syntax errors, missing includes, undeclared identifiers, and type mismatches
- Ensure all API calls use the correct signatures and argument types
- Verify that all template parameters and type casts are correct
- Do NOT attempt performance optimizations until the code compiles cleanly"""

STRATEGY_RUNTIME_ERROR = """Optimization Strategy — RUNTIME ERROR FIX:
The current implementation compiled but CRASHED at runtime. Focus entirely on fixing the crash.
- Analyze the runtime error message and stack trace in the logs above
- Check for out-of-bounds memory accesses (incorrect index calculations, buffer sizes)
- Verify kernel launch configurations (grid/block dimensions vs. data size)
- Check for illegal memory accesses and null pointer dereferences
- Ensure proper synchronization between host and device
- Validate all pointer arithmetic and array indexing
- Do NOT attempt performance optimizations until the code runs without errors"""

STRATEGY_INCORRECT_SHAPE = """Optimization Strategy — INCORRECT OUTPUT SHAPE FIX:
The implementation runs but produces OUTPUT WITH WRONG SHAPE. Focus on fixing the shape mismatch.
- Compare the expected output shapes from the specification with the actual output shapes
- Check output tensor allocation dimensions
- Verify reshape, view, transpose, and permute operations
- Ensure reduction operations use the correct axes and keepdim settings
- Check that broadcasting rules are applied correctly
- Verify loop bounds and grid dimensions match the output shape requirements
- Do NOT attempt performance optimizations until the output shapes are correct"""

STRATEGY_INCORRECT_DTYPE = """Optimization Strategy — INCORRECT OUTPUT DTYPE FIX:
The implementation runs but produces OUTPUT WITH WRONG DATA TYPE. Focus on fixing the dtype mismatch.
- Check the expected output dtypes from the specification
- Ensure output tensors are allocated with the correct dtype
- Verify type casting and conversion operations throughout the kernel
- Check that intermediate computations preserve the required precision
- Ensure accumulator types are appropriate and final results are cast to the correct type
- Do NOT attempt performance optimizations until the output dtypes are correct"""

STRATEGY_INCORRECT_NUMERICAL = """Optimization Strategy — NUMERICAL ACCURACY FIX:
The implementation runs and produces correct shapes/dtypes but has NUMERICAL ERRORS.
- Review the max relative error and max absolute error in the logs above
- Check the mathematical algorithm for correctness against the reference implementation
- Look for precision loss in reductions (use higher-precision accumulators if needed)
- Verify the order of floating-point operations (associativity matters)
- Check for missing or incorrect normalization, scaling, or bias terms
- Verify edge cases: zero values, negative values, very large/small values
- Ensure correct handling of special float values (inf, nan, denormals)
- Consider using Kahan summation or compensated algorithms for large reductions
- Do NOT attempt performance optimizations until numerical accuracy matches the reference"""

STRATEGY_TIMEOUT = """Optimization Strategy — TIMEOUT FIX:
The implementation did not complete within the time limit. Focus on making it finish in time.
- Check for infinite loops or deadlocks in the kernel
- Verify that loop bounds are correct and will terminate
- Check for synchronization deadlocks (e.g., waiting on a barrier that not all threads reach)
- Reduce unnecessary work: check if the algorithm complexity is appropriate
- Ensure kernel launch dimensions are reasonable (not launching billions of threads)
- Consider whether the problem size requires a more efficient algorithm
- Simplify the implementation first to get a working baseline, then optimize"""

STRATEGY_PASSED = """Optimization Strategy — PERFORMANCE OPTIMIZATION:
The current implementation is FUNCTIONALLY CORRECT. Focus entirely on performance optimization.
- Review the current speedup factor and latency in the logs above
- Optimize memory access patterns and coalescing for {target_gpu}
- Tune block sizes and grid dimensions for maximum occupancy
- Utilize shared memory effectively to reduce global memory transactions
- Optimize register usage and minimize divergent branches
- Leverage constant axis values for compile-time optimizations
- Consider loop unrolling, vectorized loads/stores, and instruction-level parallelism
- Minimize synchronization points and warp divergence"""

# Map EvaluationStatus to strategy prompts
ERROR_STRATEGY_MAP = {
    EvaluationStatus.COMPILE_ERROR: STRATEGY_COMPILE_ERROR,
    EvaluationStatus.RUNTIME_ERROR: STRATEGY_RUNTIME_ERROR,
    EvaluationStatus.INCORRECT_SHAPE: STRATEGY_INCORRECT_SHAPE,
    EvaluationStatus.INCORRECT_DTYPE: STRATEGY_INCORRECT_DTYPE,
    EvaluationStatus.INCORRECT_NUMERICAL: STRATEGY_INCORRECT_NUMERICAL,
    EvaluationStatus.TIMEOUT: STRATEGY_TIMEOUT,
    EvaluationStatus.PASSED: STRATEGY_PASSED,
}


def get_prompt(
    language: str,
    definition: Definition,
    target_gpu: str = "H100",
    use_ffi: bool = True,
    rag_data: str = "",
) -> str:
    prompts = {"triton": TRITON_PROMPT, "python": PYTHON_PROMPT, "cuda": CUDA_PROMPT}

    if language not in prompts:
        raise ValueError(f"Unsupported language: {language}")

    definition_str = format_definition(definition)

    # Format RAG data section
    rag_section = ""
    if rag_data:
        rag_section = f"Relevant Reference Documentation and Examples:\n{rag_data}"

    base_prompt = prompts[language].format(
        definition=definition_str, target_gpu=target_gpu, rag_data=rag_section
    )

    if language.lower() == "cuda":
        binding_prompt = FFI_PROMPT_SIMPLE if use_ffi else TORCH_BINDINGS_PROMPT
        base_prompt = base_prompt + "\n\n" + binding_prompt

    return base_prompt


def get_optimization_prompt(
    language: str,
    definition,
    trace: Trace,
    current_code: str,
    target_gpu: str = "H100",
    use_ffi: bool = True,
    rag_data: str = "",
) -> str:
    optimization_prompts = {"triton": TRITON_OPTIMIZATION_PROMPT, "cuda": CUDA_OPTIMIZATION_PROMPT}

    if language not in optimization_prompts:
        raise ValueError(f"Unsupported language for optimization: {language}")

    # Determine error type and select appropriate strategy
    if trace.evaluation is not None:
        status = trace.evaluation.status
    else:
        status = EvaluationStatus.RUNTIME_ERROR  # fallback for workload-only traces

    error_strategy = ERROR_STRATEGY_MAP.get(status, STRATEGY_RUNTIME_ERROR)
    # Format target_gpu into strategy if it has the placeholder (STRATEGY_PASSED uses it)
    error_strategy = error_strategy.format(target_gpu=target_gpu) if "{target_gpu}" in error_strategy else error_strategy

    definition_str = format_definition(definition)
    trace_logs = format_trace_logs(trace)

    # Format RAG data section
    rag_section = ""
    if rag_data:
        rag_section = f"Relevant Reference Documentation and Examples:\n{rag_data}"

    base_prompt = optimization_prompts[language].format(
        definition=definition_str,
        trace_logs=trace_logs,
        current_code=current_code,
        target_gpu=target_gpu,
        error_strategy=error_strategy,
        rag_data=rag_section,
    )

    if language.lower() == "cuda":
        binding_prompt = FFI_PROMPT_SIMPLE if use_ffi else TORCH_BINDINGS_PROMPT
        base_prompt = base_prompt + "\n\n" + binding_prompt

    return base_prompt

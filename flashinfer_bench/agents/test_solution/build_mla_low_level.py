"""
Low-level TVM-FFI compilation example for MLA kernel.

This demonstrates how to use tvm_ffi.cpp.build() directly without Solution/Definition abstraction.
Shows the raw compilation process from CUDA files to executable module.
"""

import logging
import tempfile
from pathlib import Path

import torch

logger = logging.getLogger(__name__)


def load_cuda_sources() -> dict[str, str]:
    """Load CUDA and C++ source files."""
    kernel_dir = Path(__file__).parent / "MLA_Kernel"
    
    sources = {}
    for file_path in kernel_dir.glob("*"):
        if file_path.suffix in [".cu", ".cc", ".cpp", ".h"]:
            sources[file_path.name] = file_path.read_text()
            logger.info(f"Loaded: {file_path.name}")
    
    if not sources:
        raise FileNotFoundError(f"No source files found in {kernel_dir}")
    
    return sources


def build_mla_with_tvm_ffi(
    sources: dict[str, str],
    output_dir: Path,
    kernel_name: str = "mla_kernel",
) -> tuple[str, str]:
    """
    Build MLA kernel directly using tvm_ffi.cpp.build().
    
    This is the low-level API that bypasses Solution/Definition abstraction.
    
    Parameters
    ----------
    sources : dict[str, str]
        Source code as {filename: content}
    output_dir : Path
        Directory to write sources and build artifacts
    kernel_name : str
        Name for the compiled module
        
    Returns
    -------
    tuple
        (library_path, entry_symbol)
    """
    try:
        import tvm_ffi
    except ImportError:
        raise RuntimeError("tvm_ffi is not installed. Install with: pip install tvm-ffi")
    
    logger.info("=" * 80)
    logger.info("Low-Level TVM-FFI Build Process")
    logger.info("=" * 80)
    
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Write source files to disk
    logger.info("\n[Step 1] Writing source files")
    source_paths = {}
    cpp_files = []
    cuda_files = []
    
    for filename, content in sources.items():
        file_path = output_dir / filename
        file_path.write_text(content)
        source_paths[filename] = file_path
        logger.info(f"  Wrote: {filename}")
        
        if filename.endswith(".cu"):
            cuda_files.append(str(file_path))
        elif filename.endswith((".cpp", ".cc", ".cxx", ".c")):
            cpp_files.append(str(file_path))
    
    logger.info(f"\n  CPP files: {len(cpp_files)}")
    logger.info(f"  CUDA files: {len(cuda_files)}")
    
    if not cpp_files and not cuda_files:
        raise ValueError("No C++ or CUDA source files found")
    
    # Compile with tvm_ffi.cpp.build()
    logger.info("\n[Step 2] Compiling with tvm_ffi.cpp.build()")
    logger.info(f"  Kernel name: {kernel_name}")
    logger.info(f"  Build directory: {output_dir}")
    
    try:
        library_path = tvm_ffi.cpp.build(
            name=kernel_name,
            cpp_files=cpp_files,
            cuda_files=cuda_files,
            extra_include_paths=[str(output_dir)],
            build_directory=output_dir,
        )
        logger.info(f"✓ Compilation successful!")
        logger.info(f"  Library: {library_path}")
    except Exception as e:
        logger.error(f"✗ Compilation failed: {e}")
        raise
    
    # Determine entry point (symbol)
    entry_symbol = "kernel_main"  # Adjust based on your source code
    
    return library_path, entry_symbol


def load_and_call_kernel(
    library_path: str,
    entry_symbol: str,
    batch_size: int = 1,
    seq_len: int = 128,
    hidden_dim: int = 256,
    device: str = "cuda:0",
) -> dict:
    """
    Load compiled library and call the kernel.
    
    Parameters
    ----------
    library_path : str
        Path to compiled .so file
    entry_symbol : str
        Function symbol to call
    batch_size : int
        Batch size
    seq_len : int
        Sequence length
    hidden_dim : int
        Hidden dimension
    device : str
        CUDA device
        
    Returns
    -------
    dict
        Execution results
    """
    try:
        import tvm_ffi
    except ImportError:
        raise RuntimeError("tvm_ffi is not installed")
    
    logger.info("\n" + "=" * 80)
    logger.info("Loading and Calling Kernel")
    logger.info("=" * 80)
    
    # Load the compiled module
    logger.info(f"\n[Step 3] Loading compiled module")
    logger.info(f"  Library: {library_path}")
    logger.info(f"  Entry symbol: {entry_symbol}")
    
    try:
        mod = tvm_ffi.load_module(library_path)
        logger.info("✓ Module loaded")
    except Exception as e:
        logger.error(f"✗ Failed to load module: {e}")
        raise
    
    # Get the kernel function
    try:
        kernel_fn = getattr(mod, entry_symbol)
        logger.info(f"✓ Found kernel function: {entry_symbol}")
    except AttributeError as e:
        logger.error(f"✗ Kernel function '{entry_symbol}' not found in module")
        raise
    
    # Prepare input tensors
    logger.info(f"\n[Step 4] Preparing input tensors")
    logger.info(f"  Shape: batch={batch_size}, seq_len={seq_len}, hidden={hidden_dim}")
    
    torch.manual_seed(42)
    query = torch.randn(batch_size, seq_len, hidden_dim, dtype=torch.float16, device=device)
    key = torch.randn(batch_size, seq_len, hidden_dim, dtype=torch.float16, device=device)
    value = torch.randn(batch_size, seq_len, hidden_dim, dtype=torch.float16, device=device)
    output = torch.empty(batch_size, seq_len, hidden_dim, dtype=torch.float16, device=device)
    
    logger.info(f"  query: {query.shape} {query.dtype}")
    logger.info(f"  key: {key.shape} {key.dtype}")
    logger.info(f"  value: {value.shape} {value.dtype}")
    logger.info(f"  output: {output.shape} {output.dtype}")
    
    # Warmup runs
    logger.info(f"\n[Step 5] Running warmup iterations")
    try:
        with torch.no_grad():
            for i in range(3):
                kernel_fn(query, key, value, output)
                if (i + 1) % 1 == 0:
                    logger.info(f"  Warmup {i + 1}/3 complete")
        
        torch.cuda.synchronize(device)
        logger.info("✓ Warmup complete")
    except Exception as e:
        logger.error(f"✗ Warmup failed: {e}")
        raise
    
    # Timed execution
    logger.info(f"\n[Step 6] Measuring performance (10 iterations)")
    
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    
    import time
    start = time.perf_counter()
    
    with torch.no_grad():
        for _ in range(10):
            kernel_fn(query, key, value, output)
    
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start
    
    peak_memory = torch.cuda.max_memory_allocated(device) / 1e9  # GB
    time_per_call = elapsed / 10 * 1000  # ms
    
    logger.info(f"✓ Execution complete")
    logger.info(f"  Time per call: {time_per_call:.3f} ms")
    logger.info(f"  Peak memory: {peak_memory:.3f} GB")
    logger.info(f"  Total time: {elapsed:.3f} s")
    
    results = {
        "success": True,
        "time_ms_per_call": time_per_call,
        "peak_memory_gb": peak_memory,
        "total_time_s": elapsed,
        "output_shape": tuple(output.shape),
        "output_dtype": str(output.dtype),
    }
    
    return results


def main_low_level_build():
    """Run the complete low-level TVM-FFI build and test."""
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    
    logger.info("Starting Low-Level TVM-FFI Build Example")
    logger.info("This demonstrates direct use of tvm_ffi.cpp.build() without Solution/Definition")
    
    # Load sources
    try:
        sources = load_cuda_sources()
    except Exception as e:
        logger.error(f"Failed to load sources: {e}")
        return
    
    # Create temporary build directory
    with tempfile.TemporaryDirectory(prefix="mla_tvm_") as tmpdir:
        output_dir = Path(tmpdir)
        
        # Compile with tvm_ffi
        try:
            library_path, entry_symbol = build_mla_with_tvm_ffi(
                sources,
                output_dir,
                kernel_name="mla_kernel"
            )
        except Exception as e:
            logger.error(f"Build failed: {e}")
            return
        
        # Load and test
        try:
            results = load_and_call_kernel(
                library_path,
                entry_symbol,
                batch_size=1,
                seq_len=128,
                hidden_dim=256,
                device="cuda:0"
            )
        except Exception as e:
            logger.error(f"Execution failed: {e}")
            return
        
        # Print summary
        logger.info("\n" + "=" * 80)
        logger.info("EXECUTION SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Status: {'✓ SUCCESS' if results['success'] else '✗ FAILED'}")
        logger.info(f"Time per call: {results['time_ms_per_call']:.3f} ms")
        logger.info(f"Peak memory: {results['peak_memory_gb']:.3f} GB")
        logger.info(f"Output shape: {results['output_shape']}")
        logger.info(f"Output dtype: {results['output_dtype']}")
        logger.info("=" * 80)


if __name__ == "__main__":
    main_low_level_build()

# 24.1. Unified Memory Introduction

## 24.1. Unified Memory Introduction[](#unified-memory-introduction "Permalink to this headline")

CUDA Unified Memory provides all processors with:

* A single *unified* memory pool, that is,
  a single pointer value enables all processors in the system
  (all CPUs, all GPUs, etc.)
  to access this memory with all of their native memory operations
  (pointer dereferences, atomics, etc.).
* Concurrent access to the unified memory pool from all processors in the system.

Unified Memory improves GPU programming in several ways:

* **Productivity**: GPU programs may access Unified Memory from GPU and CPU threads
  concurrently without needing to create separate allocations (`cudaMalloc()`) and
  copy memory manually back and forth (`cudaMemcpy*()`).
* **Performance**:

  + Data access speed may be maximized by migrating data towards processors that access it most frequently.
    Applications can trigger manual migration of data and may use hints to control migration heuristics.
  + Total system memory usage may be reduced by avoiding duplicating memory on both CPUs and GPUs.
* **Functionality**: It enables GPU programs to work on data that exceeds the GPU memory’s capacity.

With CUDA Unified Memory, data movement still takes place, and hints may improve performance.
These hints are not required for correctness or functionality, that is, programmers may focus on parallelizing
their applications across GPUs and CPUs first, and worry about data-movement later in the development cycle as a performance optimization.
Note that the physical location of data is invisible to a program and may be changed at any time,
but accesses to the data’s virtual address will remain valid and coherent from any processor regardless of locality.

There are two main ways to obtain CUDA Unified Memory:

* [System-Allocated Memory](#um-implicit-allocation): memory allocated on the host with system APIs:
  stack variables, global-/file-scope variables, `malloc()` / `mmap()`
  (see [System-Allocated Memory: in-depth examples](#um-system-allocator) for in-depth examples), thread locals, etc.
* [CUDA APIs that explicitly allocate Unified Memory](#um-explicit-allocation): memory allocated with, for example, `cudaMallocManaged()`,
  are available on more systems and may perform better than System-Allocated Memory.
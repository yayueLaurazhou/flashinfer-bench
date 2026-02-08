# 20.10.3. Shared Memory

### 20.10.3. Shared Memory[](#shared-memory-12-0 "Permalink to this headline")

The amount of the unified data cache reserved for shared memory is configurable on a per kernel basis and is identical to [compute capability 9.0](#shared-memory-9-0). The unified data cache has a size of 100 KB for devices of compute capability 12.0. The shared memory capacity can be set to 0, 8, 16, 32, 64, or 100 KB.

As with the [NVIDIA Ampere GPU architecture](#shared-memory-8-x), an application can configure its preferred shared memory capacity, i.e., the `carveout`. Devices of compute capability 12.0 allow a single thread block to address up to 99 KB of shared memory. Kernels relying on shared memory allocations over 48 KB per block are architecture-specific, and must use dynamic shared memory rather than statically sized shared memory arrays. These kernels require an explicit opt-in by using `cudaFuncSetAttribute()` to set the `cudaFuncAttributeMaxDynamicSharedMemorySize`; see [Shared Memory](#shared-memory-7-x) for the Volta architecture.

Note that the maximum amount of shared memory per thread block is smaller than the maximum shared memory partition available per SM. The 1 KB of shared memory not made available to a thread block is reserved for system use.
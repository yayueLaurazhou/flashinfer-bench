# Chapter 5. Rules for version mixing

1. Starting with CUDA 11.0, the ABI version for the CUDA runtime is bumped every major release.
CUDA-defined types, whether opaque handles or structures like cudaDeviceProp, have their
ABI tied to the major release of the CUDA runtime. It is unsafe to pass them from function A to
function B if those functions have been compiled with different major versions of the toolkit and
linked together into the same device executable.

2. The CUDA Driver API has a per-function ABI denoted with a _v* extension. CUDA-defined types
(e.g structs) should not be passed across different ABI versions. For example, an application calling
cuMemcpy2D_v2(const CUDA_MEMCPY2D_v2 *pCopy) and using the older version of the
struct CUDA_MEMCPY2D_v1 instead of CUDA_MEMCPY2D_v2.

3. Users should not arbitrarily mix different API versions during the lifetime of a resource. These
resources include IPC handles, memory, streams, contexts, events, etc. For example, a user
who wants to allocate CUDA memory using cuMemAlloc_v2 should free the memory using
cuMemFree_v2 and not cuMemFree.


CUDA Runtime API vRelease Version  |  8
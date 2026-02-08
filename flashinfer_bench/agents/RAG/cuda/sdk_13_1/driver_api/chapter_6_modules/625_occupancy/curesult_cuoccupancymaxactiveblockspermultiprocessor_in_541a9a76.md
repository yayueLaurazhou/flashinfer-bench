# CUresult cuOccupancyMaxActiveBlocksPerMultiprocessor (int *numBlocks, CUfunction func, int blockSize, size_t dynamicSMemSize)

Returns occupancy of a function.

###### Parameters

**numBlocks**

  - Returned occupancy
**func**

  - Kernel for which occupancy is calculated
**blockSize**

  - Block size the kernel is intended to be launched with
**dynamicSMemSize**

  - Per-block dynamic shared memory usage intended, in bytes

###### Returns

CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE,
CUDA_ERROR_UNKNOWN

###### Description

Returns in *numBlocks the number of the maximum active blocks per streaming multiprocessor.

Note that the API can also be used with context-less kernel CUkernel by querying the handle using
cuLibraryGetKernel() and then passing it to the API by casting to CUfunction. Here, the context to use
for calculations will be the current context.


Note:


Note that this function may also return error codes from previous, asynchronous launches.


See also:


CUDA Driver API TRM-06703-001 _vRelease Version  |  498


Modules


cudaOccupancyMaxActiveBlocksPerMultiprocessor
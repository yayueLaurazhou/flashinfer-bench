# CUresult cuOccupancyAvailableDynamicSMemPerBlock (size_t *dynamicSmemSize, CUfunction func, int numBlocks, int blockSize)

Returns dynamic shared memory available per block when launching numBlocks blocks on SM.

###### Parameters

**dynamicSmemSize**

  - Returned maximum dynamic shared memory
**func**

  - Kernel function for which occupancy is calculated
**numBlocks**

  - Number of blocks to fit on SM
**blockSize**

  - Size of the blocks

###### Returns

CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE,
CUDA_ERROR_UNKNOWN

###### Description

Returns in *dynamicSmemSize the maximum size of dynamic shared memory to allow
numBlocks blocks per SM.


CUDA Driver API TRM-06703-001 _vRelease Version  |  497


Modules


Note that the API can also be used with context-less kernel CUkernel by querying the handle using
cuLibraryGetKernel() and then passing it to the API by casting to CUfunction. Here, the context to use
for calculations will be the current context.


Note:


Note that this function may also return error codes from previous, asynchronous launches.
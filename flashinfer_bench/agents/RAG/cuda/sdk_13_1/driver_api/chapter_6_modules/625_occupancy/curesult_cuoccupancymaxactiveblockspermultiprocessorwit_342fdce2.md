# CUresult cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags (int *numBlocks, CUfunction func, int blockSize, size_t dynamicSMemSize, unsigned int flags)

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
**flags**

  - Requested behavior for the occupancy calculator

###### Returns

CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE,
CUDA_ERROR_UNKNOWN

###### Description

Returns in *numBlocks the number of the maximum active blocks per streaming multiprocessor.

The Flags parameter controls how special cases are handled. The valid flags are:

CU_OCCUPANCY_DEFAULT, which maintains the default behavior as

###### **‣**

cuOccupancyMaxActiveBlocksPerMultiprocessor;

CU_OCCUPANCY_DISABLE_CACHING_OVERRIDE, which suppresses the default

###### **‣**

behavior on platform where global caching affects occupancy. On such platforms, if
caching is enabled, but per-block SM resource usage would result in zero occupancy,
the occupancy calculator will calculate the occupancy as if caching is disabled. Setting
CU_OCCUPANCY_DISABLE_CACHING_OVERRIDE makes the occupancy calculator to
return 0 in such cases. More information can be found about this feature in the "Unified L1/Texture
Cache" section of the Maxwell tuning guide.


CUDA Driver API TRM-06703-001 _vRelease Version  |  499


Modules


Note that the API can also be with launch context-less kernel CUkernel by querying the handle using
cuLibraryGetKernel() and then passing it to the API by casting to CUfunction. Here, the context to use
for calculations will be the current context.


Note:


Note that this function may also return error codes from previous, asynchronous launches.


See also:

cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags
# CUresult cuOccupancyMaxPotentialBlockSize (int *minGridSize, int *blockSize, CUfunction func, CUoccupancyB2DSize blockSizeToDynamicSMemSize, size_t dynamicSMemSize, int blockSizeLimit)

Suggest a launch configuration with reasonable occupancy.

###### Parameters

**minGridSize**

  - Returned minimum grid size needed to achieve the maximum occupancy
**blockSize**

  - Returned maximum block size that can achieve the maximum occupancy
**func**

  - Kernel for which launch configuration is calculated
**blockSizeToDynamicSMemSize**

  - A function that calculates how much per-block dynamic shared memory func uses based on the
block size
**dynamicSMemSize**

  - Dynamic shared memory usage intended, in bytes
**blockSizeLimit**

  - The maximum block size func is designed to handle

###### Returns

CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE,
CUDA_ERROR_UNKNOWN


CUDA Driver API TRM-06703-001 _vRelease Version  |  501


Modules

###### Description

Returns in *blockSize a reasonable block size that can achieve the maximum occupancy (or, the
maximum number of active warps with the fewest blocks per multiprocessor), and in *minGridSize
the minimum grid size to achieve the maximum occupancy.

If blockSizeLimit is 0, the configurator will use the maximum block size permitted by the device /
function instead.

If per-block dynamic shared memory allocation is not needed, the user should leave both
blockSizeToDynamicSMemSize and dynamicSMemSize as 0.

If per-block dynamic shared memory allocation is needed, then if the dynamic shared memory size
is constant regardless of block size, the size should be passed through dynamicSMemSize, and
blockSizeToDynamicSMemSize should be NULL.

Otherwise, if the per-block dynamic shared memory size varies with different block sizes, the user
needs to provide a unary function through blockSizeToDynamicSMemSize that computes the
dynamic shared memory needed by func for any given block size. dynamicSMemSize is ignored.
An example signature is:

‎  // Take block size, returns dynamic shared memory needed
size_t blockToSmem(int blockSize);

Note that the API can also be used with context-less kernel CUkernel by querying the handle using
cuLibraryGetKernel() and then passing it to the API by casting to CUfunction. Here, the context to use
for calculations will be the current context.


Note:


Note that this function may also return error codes from previous, asynchronous launches.


See also:

cudaOccupancyMaxPotentialBlockSize


CUDA Driver API TRM-06703-001 _vRelease Version  |  502


Modules
# CUresult cuOccupancyMaxPotentialBlockSizeWithFlags (int *minGridSize, int *blockSize, CUfunction func, CUoccupancyB2DSize blockSizeToDynamicSMemSize, size_t dynamicSMemSize, int blockSizeLimit, unsigned int flags)

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
**flags**

  - Options

###### Returns

CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE,
CUDA_ERROR_UNKNOWN

###### Description

An extended version of cuOccupancyMaxPotentialBlockSize. In addition to arguments passed to
cuOccupancyMaxPotentialBlockSize, cuOccupancyMaxPotentialBlockSizeWithFlags also takes a
Flags parameter.

The Flags parameter controls how special cases are handled. The valid flags are:

CU_OCCUPANCY_DEFAULT, which maintains the default behavior as

###### **‣**

cuOccupancyMaxPotentialBlockSize;

CU_OCCUPANCY_DISABLE_CACHING_OVERRIDE, which suppresses the default

###### **‣**

behavior on platform where global caching affects occupancy. On such platforms, the launch


CUDA Driver API TRM-06703-001 _vRelease Version  |  503


Modules


configurations that produces maximal occupancy might not support global caching. Setting
CU_OCCUPANCY_DISABLE_CACHING_OVERRIDE guarantees that the the produced launch
configuration is global caching compatible at a potential cost of occupancy. More information can
be found about this feature in the "Unified L1/Texture Cache" section of the Maxwell tuning guide.

Note that the API can also be used with context-less kernel CUkernel by querying the handle using
cuLibraryGetKernel() and then passing it to the API by casting to CUfunction. Here, the context to use
for calculations will be the current context.


Note:


Note that this function may also return error codes from previous, asynchronous launches.


See also:

cudaOccupancyMaxPotentialBlockSizeWithFlags
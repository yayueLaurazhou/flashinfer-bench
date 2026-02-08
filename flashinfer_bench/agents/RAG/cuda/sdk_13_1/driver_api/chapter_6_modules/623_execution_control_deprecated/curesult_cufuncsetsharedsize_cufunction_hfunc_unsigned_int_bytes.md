# CUresult cuFuncSetSharedSize (CUfunction hfunc, unsigned int bytes)

Sets the dynamic shared-memory size for the function.

###### Parameters

**hfunc**

  - Kernel to specify dynamic shared-memory size for
**bytes**

  - Dynamic shared-memory size per thread in bytes

###### Returns

CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_HANDLE,
CUDA_ERROR_INVALID_VALUE

###### Description

Deprecated

Sets through bytes the amount of dynamic shared memory that will be available to each thread block
when the kernel given by hfunc is launched.


Note:


Note that this function may also return error codes from previous, asynchronous launches.


See also:

cuFuncSetBlockShape, cuFuncSetCacheConfig, cuFuncGetAttribute, cuParamSetSize, cuParamSeti,
cuParamSetf, cuParamSetv, cuLaunch, cuLaunchGrid, cuLaunchGridAsync, cuLaunchKernel


CUDA Driver API TRM-06703-001 _vRelease Version  |  406


Modules
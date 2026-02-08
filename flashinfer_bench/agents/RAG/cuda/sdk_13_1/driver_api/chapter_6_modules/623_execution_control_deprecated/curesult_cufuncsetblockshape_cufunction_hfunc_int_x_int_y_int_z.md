# CUresult cuFuncSetBlockShape (CUfunction hfunc, int x, int y, int z)

Sets the block-dimensions for the function.

###### Parameters

**hfunc**

  - Kernel to specify dimensions of
**x**

  - X dimension
**y**

  - Y dimension
**z**

  - Z dimension

###### Returns

CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_HANDLE,
CUDA_ERROR_INVALID_VALUE

###### Description

Deprecated

Specifies the x, y, and z dimensions of the thread blocks that are created when the kernel given by
hfunc is launched.


Note:


Note that this function may also return error codes from previous, asynchronous launches.


See also:

cuFuncSetSharedSize, cuFuncSetCacheConfig, cuFuncGetAttribute, cuParamSetSize, cuParamSeti,
cuParamSetf, cuParamSetv, cuLaunch, cuLaunchGrid, cuLaunchGridAsync, cuLaunchKernel


CUDA Driver API TRM-06703-001 _vRelease Version  |  404


Modules
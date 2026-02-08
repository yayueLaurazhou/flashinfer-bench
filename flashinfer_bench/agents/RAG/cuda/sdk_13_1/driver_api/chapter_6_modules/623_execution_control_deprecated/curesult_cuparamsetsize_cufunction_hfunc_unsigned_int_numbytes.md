# CUresult cuParamSetSize (CUfunction hfunc, unsigned int numbytes)

Sets the parameter size for the function.

###### Parameters

**hfunc**

  - Kernel to set parameter size for
**numbytes**

  - Size of parameter list in bytes


CUDA Driver API TRM-06703-001 _vRelease Version  |  411


Modules

###### Returns

CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE

###### Description

Deprecated

Sets through numbytes the total size in bytes needed by the function parameters of the kernel
corresponding to hfunc.


Note:


Note that this function may also return error codes from previous, asynchronous launches.


See also:

cuFuncSetBlockShape, cuFuncSetSharedSize, cuFuncGetAttribute, cuParamSetf, cuParamSeti,
cuParamSetv, cuLaunch, cuLaunchGrid, cuLaunchGridAsync, cuLaunchKernel
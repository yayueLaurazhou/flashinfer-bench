# CUresult cuParamSetf (CUfunction hfunc, int offset, float value)

Adds a floating-point parameter to the function's argument list.

###### Parameters

**hfunc**

  - Kernel to add parameter to
**offset**

  - Offset to add parameter to argument list
**value**

  - Value of parameter

###### Returns

CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE

###### Description

Deprecated

Sets a floating-point parameter that will be specified the next time the kernel corresponding to hfunc
will be invoked. offset is a byte offset.


Note:


Note that this function may also return error codes from previous, asynchronous launches.


See also:

cuFuncSetBlockShape, cuFuncSetSharedSize, cuFuncGetAttribute, cuParamSetSize, cuParamSeti,
cuParamSetv, cuLaunch, cuLaunchGrid, cuLaunchGridAsync, cuLaunchKernel


CUDA Driver API TRM-06703-001 _vRelease Version  |  410


Modules
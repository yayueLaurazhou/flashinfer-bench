# CUresult cuParamSetv (CUfunction hfunc, int offset, void *ptr, unsigned int numbytes)

Adds arbitrary data to the function's argument list.

###### Parameters

**hfunc**

  - Kernel to add data to
**offset**

  - Offset to add data to argument list
**ptr**

  - Pointer to arbitrary data
**numbytes**

  - Size of data to copy in bytes

###### Returns

CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE

###### Description

Deprecated

Copies an arbitrary amount of data (specified in numbytes) from ptr into the parameter space of the
kernel corresponding to hfunc. offset is a byte offset.


Note:


Note that this function may also return error codes from previous, asynchronous launches.


See also:

cuFuncSetBlockShape, cuFuncSetSharedSize, cuFuncGetAttribute, cuParamSetSize, cuParamSetf,
cuParamSeti, cuLaunch, cuLaunchGrid, cuLaunchGridAsync, cuLaunchKernel


CUDA Driver API TRM-06703-001 _vRelease Version  |  413


Modules
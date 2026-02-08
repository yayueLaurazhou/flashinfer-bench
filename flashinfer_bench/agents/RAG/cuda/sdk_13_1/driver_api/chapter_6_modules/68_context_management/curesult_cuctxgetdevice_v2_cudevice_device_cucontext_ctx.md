# CUresult cuCtxGetDevice_v2 (CUdevice *device, CUcontext ctx)

Returns the device handle for the specified context.

###### Parameters

**device**

  - Returned device handle for the specified context
**ctx**

  - Context for which to obtain the device

###### Returns

CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE

###### Description

Returns in *device the handle of the specified context's device. If the specified context is NULL, the
API will return the current context's device.


Note:


Note that this function may also return error codes from previous, asynchronous launches.


See also:

cuCtxGetCurrent, cuCtxPopCurrent, cuCtxPushCurrent
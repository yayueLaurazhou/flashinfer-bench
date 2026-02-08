# CUresult cuCtxGetDevice (CUdevice *device)

Returns the device handle for the current context.

###### Parameters

**device**

  - Returned device handle for the current context

###### Returns

CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE,

###### Description

Returns in *device the handle of the current context's device.


Note:


Note that this function may also return error codes from previous, asynchronous launches.


CUDA Driver API TRM-06703-001 _vRelease Version  |  126


Modules


See also:

cuCtxCreate, cuCtxDestroy, cuCtxGetApiVersion, cuCtxGetCacheConfig, cuCtxGetFlags,
cuCtxGetLimit, cuCtxPopCurrent, cuCtxPushCurrent, cuCtxSetCacheConfig, cuCtxSetLimit,
cuCtxSynchronize, cudaGetDevice
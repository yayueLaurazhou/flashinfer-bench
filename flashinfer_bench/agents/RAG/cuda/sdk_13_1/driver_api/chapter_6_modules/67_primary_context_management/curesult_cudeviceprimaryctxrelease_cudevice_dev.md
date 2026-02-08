# CUresult cuDevicePrimaryCtxRelease (CUdevice dev)

Release the primary context on the GPU.

###### Parameters

**dev**

  - Device which primary context is released

###### Returns

CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
CUDA_ERROR_INVALID_DEVICE, CUDA_ERROR_INVALID_CONTEXT

###### Description

Releases the primary context interop on the device. A retained context should always be released once
the user is done using it. The context is automatically reset once the last reference to it is released. This
behavior is different when the primary context was retained by the CUDA runtime from CUDA 4.0 and
earlier. In this case, the primary context remains always active.

Releasing a primary context that has not been previously retained will fail with
CUDA_ERROR_INVALID_CONTEXT.

Please note that unlike cuCtxDestroy() this method does not pop the context from stack in any
circumstances.


Note:


CUDA Driver API TRM-06703-001 _vRelease Version  |  115


Modules


Note that this function may also return error codes from previous, asynchronous launches.


See also:

cuDevicePrimaryCtxRetain, cuCtxDestroy, cuCtxGetApiVersion, cuCtxGetCacheConfig,
cuCtxGetDevice, cuCtxGetFlags, cuCtxGetLimit, cuCtxPopCurrent, cuCtxPushCurrent,
cuCtxSetCacheConfig, cuCtxSetLimit, cuCtxSynchronize
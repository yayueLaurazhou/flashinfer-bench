# CUresult cuCtxPushCurrent (CUcontext ctx)

Pushes a context on the current CPU thread.

###### Parameters

**ctx**

  - Context to push

###### Returns

CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE

###### Description

Pushes the given context ctx onto the CPU thread's stack of current contexts. The specified context
becomes the CPU thread's current context, so all CUDA functions that operate on the current context
are affected.

The previous current context may be made current again by calling cuCtxDestroy() or
cuCtxPopCurrent().


Note:


Note that this function may also return error codes from previous, asynchronous launches.


See also:

cuCtxCreate, cuCtxDestroy, cuCtxGetApiVersion, cuCtxGetCacheConfig, cuCtxGetDevice,
cuCtxGetFlags, cuCtxGetLimit, cuCtxPopCurrent, cuCtxSetCacheConfig, cuCtxSetLimit,
cuCtxSynchronize


CUDA Driver API TRM-06703-001 _vRelease Version  |  132


Modules
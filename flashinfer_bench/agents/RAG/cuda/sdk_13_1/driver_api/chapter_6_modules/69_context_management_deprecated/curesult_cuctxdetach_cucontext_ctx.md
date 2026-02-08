# CUresult cuCtxDetach (CUcontext ctx)

Decrement a context's usage-count.

###### Parameters

**ctx**

  - Context to destroy


CUDA Driver API TRM-06703-001 _vRelease Version  |  141


Modules

###### Returns

CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
CUDA_ERROR_INVALID_CONTEXT

###### Description

Deprecated

Note that this function is deprecated and should not be used.

Decrements the usage count of the context ctx, and destroys the context if the usage count goes to 0.
The context must be a handle that was passed back by cuCtxCreate() or cuCtxAttach(), and must be
current to the calling thread.


Note:


Note that this function may also return error codes from previous, asynchronous launches.


See also:

cuCtxCreate, cuCtxDestroy, cuCtxGetApiVersion, cuCtxGetCacheConfig, cuCtxGetDevice,
cuCtxGetFlags, cuCtxGetLimit, cuCtxPopCurrent, cuCtxPushCurrent, cuCtxSetCacheConfig,
cuCtxSetLimit, cuCtxSynchronize
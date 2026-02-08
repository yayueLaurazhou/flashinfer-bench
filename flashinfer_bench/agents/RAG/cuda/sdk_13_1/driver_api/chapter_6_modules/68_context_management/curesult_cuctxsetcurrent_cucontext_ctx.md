# CUresult cuCtxSetCurrent (CUcontext ctx)

Binds the specified CUDA context to the calling CPU thread.

###### Parameters

**ctx**

  - Context to bind to the calling CPU thread

###### Returns

CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
CUDA_ERROR_INVALID_CONTEXT

###### Description

Binds the specified CUDA context to the calling CPU thread. If ctx is NULL then the CUDA context
previously bound to the calling CPU thread is unbound and CUDA_SUCCESS is returned.

If there exists a CUDA context stack on the calling CPU thread, this will replace the top of that stack
with ctx. If ctx is NULL then this will be equivalent to popping the top of the calling CPU thread's
CUDA context stack (or a no-op if the calling CPU thread's CUDA context stack is empty).


Note:


Note that this function may also return error codes from previous, asynchronous launches.


See also:

cuCtxGetCurrent, cuCtxCreate, cuCtxDestroy, cudaSetDevice


CUDA Driver API TRM-06703-001 _vRelease Version  |  135


Modules
# CUresult cuCtxPopCurrent (CUcontext *pctx)

Pops the current CUDA context from the current CPU thread.

###### Parameters

**pctx**

  - Returned popped context handle

###### Returns

CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
CUDA_ERROR_INVALID_CONTEXT

###### Description

Pops the current CUDA context from the CPU thread and passes back the old context handle
in *pctx. That context may then be made current to a different CPU thread by calling
cuCtxPushCurrent().


CUDA Driver API TRM-06703-001 _vRelease Version  |  131


Modules


If a context was current to the CPU thread before cuCtxCreate() or cuCtxPushCurrent() was called, this
function makes that context current to the CPU thread again.


Note:


Note that this function may also return error codes from previous, asynchronous launches.


See also:

cuCtxCreate, cuCtxDestroy, cuCtxGetApiVersion, cuCtxGetCacheConfig, cuCtxGetDevice,
cuCtxGetFlags, cuCtxGetLimit, cuCtxPushCurrent, cuCtxSetCacheConfig, cuCtxSetLimit,
cuCtxSynchronize
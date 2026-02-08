# CUresult cuCtxAttach (CUcontext *pctx, unsigned int flags)

Increment a context's usage-count.

###### Parameters

**pctx**

  - Returned context handle of the current context
**flags**

  - Context attach flags (must be 0)

###### Returns

CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE

###### Description

Deprecated

Note that this function is deprecated and should not be used.

Increments the usage count of the context and passes back a context handle in *pctx that must be
passed to cuCtxDetach() when the application is done with the context. cuCtxAttach() fails if there is
no context current to the thread.

Currently, the flags parameter must be 0.


Note:


Note that this function may also return error codes from previous, asynchronous launches.


See also:

cuCtxCreate, cuCtxDestroy, cuCtxDetach, cuCtxGetApiVersion, cuCtxGetCacheConfig,
cuCtxGetDevice, cuCtxGetFlags, cuCtxGetLimit, cuCtxPopCurrent, cuCtxPushCurrent,
cuCtxSetCacheConfig, cuCtxSetLimit, cuCtxSynchronize
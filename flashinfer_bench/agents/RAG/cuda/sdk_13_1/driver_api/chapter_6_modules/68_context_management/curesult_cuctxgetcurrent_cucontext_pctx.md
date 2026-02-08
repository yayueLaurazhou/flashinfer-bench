# CUresult cuCtxGetCurrent (CUcontext *pctx)

Returns the CUDA context bound to the calling CPU thread.

###### Parameters

**pctx**

  - Returned context handle

###### Returns

CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,

###### Description

Returns in *pctx the CUDA context bound to the calling CPU thread. If no context is bound to the
calling CPU thread then *pctx is set to NULL and CUDA_SUCCESS is returned.


Note:


Note that this function may also return error codes from previous, asynchronous launches.


See also:

cuCtxSetCurrent, cuCtxCreate, cuCtxDestroy, cudaGetDevice
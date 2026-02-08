# CUresult cuCtxGetId (CUcontext ctx, unsigned long long *ctxId)

Returns the unique Id associated with the context supplied.

###### Parameters

**ctx**

  - Context for which to obtain the Id
**ctxId**

  - Pointer to store the Id of the context

###### Returns

CUDA_SUCCESS, CUDA_ERROR_CONTEXT_IS_DESTROYED,
CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE

###### Description

Returns in ctxId the unique Id which is associated with a given context. The Id is unique for the life
of the program for this instance of CUDA. If context is supplied as NULL and there is one current, the
Id of the current context is returned.


Note:


Note that this function may also return error codes from previous, asynchronous launches.


See also:

cuCtxCreate, cuCtxDestroy, cuCtxGetApiVersion, cuCtxGetCacheConfig, cuCtxGetDevice,
cuCtxGetFlags, cuCtxGetLimit, cuCtxPushCurrent
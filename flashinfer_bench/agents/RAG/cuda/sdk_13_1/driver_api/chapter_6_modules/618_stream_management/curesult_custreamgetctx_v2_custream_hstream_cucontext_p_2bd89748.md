# CUresult cuStreamGetCtx_v2 (CUstream hStream, CUcontext *pCtx, CUgreenCtx *pGreenCtx)

Query the contexts associated with a stream.

###### Parameters

**hStream**

  - Handle to the stream to be queried
**pCtx**

  - Returned regular context associated with the stream
**pGreenCtx**

  - Returned green context if the stream is associated with a green context or NULL if not

###### Returns

CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_HANDLE

###### Description

Returns the contexts that the stream is associated with.

If the stream is associated with a green context, the API returns the green context in pGreenCtx and
the primary context of the associated device in pCtx.

If the stream is associated with a regular context, the API returns the regular context in pCtx and
NULL in pGreenCtx.

The stream handle hStream can refer to any of the following:

a stream created via any of the CUDA driver APIs such as cuStreamCreate,

###### **‣**

cuStreamCreateWithPriority and cuGreenCtxStreamCreate, or their runtime API equivalents such


CUDA Driver API TRM-06703-001 _vRelease Version  |  344


Modules


as cudaStreamCreate, cudaStreamCreateWithFlags and cudaStreamCreateWithPriority. Passing an
invalid handle will result in undefined behavior.
any of the special streams such as the NULL stream, CU_STREAM_LEGACY and

###### **‣**

CU_STREAM_PER_THREAD. The runtime API equivalents of these are also accepted, which
are NULL, cudaStreamLegacy and cudaStreamPerThread respectively. If any of the special
handles are specified, the API will operate on the context current to the calling thread. If a green
context (that was converted via cuCtxFromGreenCtx() before setting it current) is current to the
calling thread, the API will return the green context in pGreenCtx and the primary context of
the associated device in pCtx. If a regular context is current, the API returns the regular context
in pCtx and NULL in pGreenCtx. Note that specifying CU_STREAM_PER_THREAD
or cudaStreamPerThread will return CUDA_ERROR_INVALID_HANDLE if a green
context is current to the calling thread. If no context is current to the calling thread,
CUDA_ERROR_INVALID_CONTEXT is returned.


Note:


Note that this function may also return error codes from previous, asynchronous launches.


See also:

cuStreamDestroy, cuStreamCreate cuStreamCreateWithPriority, cuGreenCtxStreamCreate,
cuStreamGetPriority, cuStreamGetFlags, cuStreamGetDevice, cuStreamWaitEvent, cuStreamQuery,
cuStreamSynchronize, cuStreamAddCallback, cudaStreamCreate, cudaStreamCreateWithFlags,
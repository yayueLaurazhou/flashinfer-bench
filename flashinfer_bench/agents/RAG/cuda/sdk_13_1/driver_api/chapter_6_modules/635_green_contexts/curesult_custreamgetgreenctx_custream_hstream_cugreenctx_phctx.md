# CUresult cuStreamGetGreenCtx (CUstream hStream, CUgreenCtx *phCtx)

Query the green context associated with a stream.

###### Parameters

**hStream**

  - Handle to the stream to be queried
**phCtx**

  - Returned green context associated with the stream

###### Returns

CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_HANDLE,


CUDA Driver API TRM-06703-001 _vRelease Version  |  589


Modules

###### Description

Returns the CUDA green context that the stream is associated with, or NULL if the stream is not
associated with any green context.

The stream handle hStream can refer to any of the following:

a stream created via any of the CUDA driver APIs such as cuStreamCreate,

###### **‣**

cuStreamCreateWithPriority and cuGreenCtxStreamCreate, or their runtime API equivalents
such as cudaStreamCreate, cudaStreamCreateWithFlags and cudaStreamCreateWithPriority.
If during stream creation the context that was active in the calling thread was obtained with
cuCtxFromGreenCtx, that green context is returned in phCtx. Otherwise, *phCtx is set to NULL
instead.
special stream such as the NULL stream or CU_STREAM_LEGACY. In that case if context that is

###### **‣**

active in the calling thread was obtained with cuCtxFromGreenCtx, that green context is returned.
Otherwise, *phCtx is set to NULL instead.

Passing an invalid handle will result in undefined behavior.


Note:


Note that this function may also return error codes from previous, asynchronous launches.


See also:

cuStreamDestroy, cuStreamCreate, cuStreamCreateWithPriority, cuStreamGetCtx,
cuGreenCtxStreamCreate, cuStreamGetPriority, cuStreamGetFlags, cuStreamGetDevice,
cuStreamWaitEvent, cuStreamQuery, cuStreamSynchronize, cuStreamAddCallback,
cudaStreamCreate, cudaStreamCreateWithFlags
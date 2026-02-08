# CUresult cuStreamGetCtx (CUstream hStream, CUcontext *pctx)

Query the context associated with a stream.

###### Parameters

**hStream**

  - Handle to the stream to be queried
**pctx**

  - Returned context associated with the stream

###### Returns

CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_HANDLE,
CUDA_ERROR_NOT_SUPPORTED

###### Description

Returns the CUDA context that the stream is associated with.

If the stream was created via the API cuGreenCtxStreamCreate, the returned context is equivalent to
the one returned by cuCtxFromGreenCtx() on the green context associated with the stream at creation
time.

The stream handle hStream can refer to any of the following:

a stream created via any of the CUDA driver APIs such as cuStreamCreate and

###### **‣**

cuStreamCreateWithPriority, or their runtime API equivalents such as cudaStreamCreate,
cudaStreamCreateWithFlags and cudaStreamCreateWithPriority. The returned context is the
context that was active in the calling thread when the stream was created. Passing an invalid handle
will result in undefined behavior.
any of the special streams such as the NULL stream, CU_STREAM_LEGACY and

###### **‣**

CU_STREAM_PER_THREAD. The runtime API equivalents of these are also accepted, which are
NULL, cudaStreamLegacy and cudaStreamPerThread respectively. Specifying any of the special


CUDA Driver API TRM-06703-001 _vRelease Version  |  343


Modules


handles will return the context current to the calling thread. If no context is current to the calling
thread, CUDA_ERROR_INVALID_CONTEXT is returned.


Note:


Note that this function may also return error codes from previous, asynchronous launches.


See also:

cuStreamDestroy, cuStreamCreateWithPriority, cuStreamGetPriority, cuStreamGetFlags,
cuStreamGetDevice cuStreamWaitEvent, cuStreamQuery, cuStreamSynchronize,
cuStreamAddCallback, cudaStreamCreate, cudaStreamCreateWithFlags
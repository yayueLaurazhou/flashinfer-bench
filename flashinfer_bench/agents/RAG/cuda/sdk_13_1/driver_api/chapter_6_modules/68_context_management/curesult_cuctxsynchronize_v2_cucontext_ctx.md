# CUresult cuCtxSynchronize_v2 (CUcontext ctx)

Block for the specified context's tasks to complete.

###### Parameters

**ctx**

  - Context to synchronize

###### Returns

CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE,
CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED

###### Description

Blocks until the specified context has completed all preceding requested tasks. If the specified context
is the primary context, green contexts that have been created will also be synchronized. The API
returns an error if one of the preceding tasks failed.

If the context was created with the CU_CTX_SCHED_BLOCKING_SYNC flag, the CPU thread will
block until the GPU context has finished its work.

If the specified context is NULL, the API will operate on the current context.


Note:


Note that this function may also return error codes from previous, asynchronous launches.


See also:

cuCtxGetCurrent, cuCtxPopCurrent, cuCtxPushCurrent, cuGreenCtxCreate, cuCtxFromGreenCtx,
cudaDeviceSynchronize


CUDA Driver API TRM-06703-001 _vRelease Version  |  139


Modules
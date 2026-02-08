# CUresult cuCtxSynchronize (void)

Block for the current context's tasks to complete.

###### Returns

CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED

###### Description

Blocks until the current context has completed all preceding requested tasks. If the current
context is the primary context, green contexts that have been created will also be synchronized.
cuCtxSynchronize() returns an error if one of the preceding tasks failed. If the context was created with
the CU_CTX_SCHED_BLOCKING_SYNC flag, the CPU thread will block until the GPU context has
finished its work.


Note:


Note that this function may also return error codes from previous, asynchronous launches.


See also:


CUDA Driver API TRM-06703-001 _vRelease Version  |  138


Modules


cuCtxCreate, cuCtxDestroy, cuCtxGetApiVersion, cuCtxGetCacheConfig, cuCtxGetDevice,
cuCtxGetFlags, cuCtxGetLimit, cuCtxPopCurrent, cuCtxPushCurrent, cuCtxSetCacheConfig,
cuCtxSetLimit, cudaDeviceSynchronize
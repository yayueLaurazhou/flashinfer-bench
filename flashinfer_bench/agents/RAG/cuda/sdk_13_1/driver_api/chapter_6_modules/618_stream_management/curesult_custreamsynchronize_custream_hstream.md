# CUresult cuStreamSynchronize (CUstream hStream)

Wait until a stream's tasks are completed.

###### Parameters

**hStream**

  - Stream to wait for

###### Returns

CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_HANDLE

###### Description

Waits until the device has completed all operations in the stream specified by hStream. If the context
was created with the CU_CTX_SCHED_BLOCKING_SYNC flag, the CPU thread will block until the
stream is finished with all of its tasks.





See also:

cuStreamCreate, cuStreamDestroy, cuStreamWaitEvent, cuStreamQuery, cuStreamAddCallback,
cudaStreamSynchronize
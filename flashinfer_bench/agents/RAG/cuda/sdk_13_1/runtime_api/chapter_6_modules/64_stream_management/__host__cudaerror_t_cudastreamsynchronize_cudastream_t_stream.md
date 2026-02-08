# __host__cudaError_t cudaStreamSynchronize (cudaStream_t stream)

Waits for stream tasks to complete.

##### Parameters

**stream**

  - Stream identifier

##### Returns

cudaSuccess, cudaErrorInvalidResourceHandle


CUDA Runtime API vRelease Version  |  68


Modules

##### Description

Blocks until stream has completed all operations. If the cudaDeviceScheduleBlockingSync flag was
set for this device, the host thread will block until the stream is finished with all of its tasks.











See also:

cudaStreamCreate, cudaStreamCreateWithFlags, cudaStreamQuery, cudaStreamWaitEvent,
cudaStreamAddCallback, cudaStreamDestroy, cuStreamSynchronize
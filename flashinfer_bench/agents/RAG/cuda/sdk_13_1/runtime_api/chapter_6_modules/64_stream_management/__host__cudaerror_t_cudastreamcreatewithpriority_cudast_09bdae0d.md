# __host__cudaError_t cudaStreamCreateWithPriority (cudaStream_t *pStream, unsigned int flags, int priority)

Create an asynchronous stream with the specified priority.

##### Parameters

**pStream**

  - Pointer to new stream identifier
**flags**

  - Flags for stream creation. See cudaStreamCreateWithFlags for a list of valid flags that can be
passed
**priority**

  - Priority of the stream. Lower numbers represent higher priorities. See
cudaDeviceGetStreamPriorityRange for more information about the meaningful stream priorities
that can be passed.

##### Returns

cudaSuccess, cudaErrorInvalidValue

##### Description

Creates a stream with the specified priority and returns a handle in pStream. The stream is created on
the context that is current to the calling host thread. If no context is current to the calling host thread,
then the primary context for a device is selected, made current to the calling thread, and initialized
before creating a stream on it. This affects the scheduling priority of work in the stream. Priorities
provide a hint to preferentially run work with higher priority when possible, but do not preempt
already-running work or provide any other functional guarantee on execution order.

priority follows a convention where lower numbers represent higher priorities. '0'
represents default priority. The range of meaningful numerical priorities can be queried using
cudaDeviceGetStreamPriorityRange. If the specified priority is outside the numerical range returned
by cudaDeviceGetStreamPriorityRange, it will automatically be clamped to the lowest or the highest
number in the range.



CUDA Runtime API vRelease Version  |  58


Modules


**‣** In the current implementation, only compute kernels launched in priority streams are affected by
the stream's priority. Stream priorities have no effect on host-to-device and device-to-host memory
operations.


See also:

cudaStreamCreate, cudaStreamCreateWithFlags, cudaDeviceGetStreamPriorityRange,
cudaStreamGetPriority, cudaStreamQuery, cudaStreamWaitEvent, cudaStreamAddCallback,
cudaStreamSynchronize, cudaSetDevice, cudaStreamDestroy, cuStreamCreateWithPriority
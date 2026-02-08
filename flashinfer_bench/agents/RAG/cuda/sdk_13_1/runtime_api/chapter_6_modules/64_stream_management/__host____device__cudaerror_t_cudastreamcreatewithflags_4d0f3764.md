# __host____device__cudaError_t cudaStreamCreateWithFlags (cudaStream_t *pStream, unsigned int flags)

Create an asynchronous stream.

##### Parameters

**pStream**

  - Pointer to new stream identifier
**flags**

  - Parameters for stream creation

##### Returns

cudaSuccess, cudaErrorInvalidValue

##### Description

Creates a new asynchronous stream on the context that is current to the calling host thread. If no
context is current to the calling host thread, then the primary context for a device is selected, made
current to the calling thread, and initialized before creating a stream on it. The flags argument
determines the behaviors of the stream. Valid values for flags are

cudaStreamDefault: Default stream creation flag.

##### **‣**

cudaStreamNonBlocking: Specifies that work running in the created stream may run concurrently

##### **‣**

with work in stream 0 (the NULL stream), and that the created stream should perform no implicit
synchronization with stream 0.





See also:

cudaStreamCreate, cudaStreamCreateWithPriority, cudaStreamGetFlags, cudaStreamGetDevice,
cudaStreamGetDevResource, cudaStreamQuery, cudaStreamSynchronize, cudaStreamWaitEvent,
cudaStreamAddCallback, cudaSetDevice, cudaStreamDestroy, cuStreamCreate


CUDA Runtime API vRelease Version  |  57


Modules
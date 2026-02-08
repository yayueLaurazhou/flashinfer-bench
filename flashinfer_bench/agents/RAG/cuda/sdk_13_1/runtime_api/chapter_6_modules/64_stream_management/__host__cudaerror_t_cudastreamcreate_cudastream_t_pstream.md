# __host__cudaError_t cudaStreamCreate (cudaStream_t *pStream)

Create an asynchronous stream.

##### Parameters

**pStream**

  - Pointer to new stream identifier

##### Returns

cudaSuccess, cudaErrorInvalidValue

##### Description

Creates a new asynchronous stream on the context that is current to the calling host thread. If no
context is current to the calling host thread, then the primary context for a device is selected, made
current to the calling thread, and initialized before creating a stream on it.





See also:

cudaStreamCreateWithPriority, cudaStreamCreateWithFlags, cudaStreamGetPriority,
cudaStreamGetFlags, cudaStreamGetDevice, cudaStreamGetDevResource, cudaStreamQuery,
cudaStreamSynchronize, cudaStreamWaitEvent, cudaStreamAddCallback, cudaSetDevice,
cudaStreamDestroy, cuStreamCreate


CUDA Runtime API vRelease Version  |  56


Modules
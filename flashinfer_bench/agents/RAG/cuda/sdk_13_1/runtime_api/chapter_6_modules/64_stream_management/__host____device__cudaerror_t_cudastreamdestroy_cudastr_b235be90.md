# __host____device__cudaError_t cudaStreamDestroy (cudaStream_t stream)

Destroys and cleans up an asynchronous stream.

##### Parameters

**stream**

  - Stream identifier

##### Returns

cudaSuccess, cudaErrorInvalidValue, cudaErrorInvalidResourceHandle

##### Description

Destroys and cleans up the asynchronous stream specified by stream.

In case the device is still doing work in the stream stream when cudaStreamDestroy() is called,
the function will return immediately and the resources associated with stream will be released
automatically once the device has completed all work in stream.











See also:


CUDA Runtime API vRelease Version  |  59


Modules


cudaStreamCreate, cudaStreamCreateWithFlags, cudaStreamQuery, cudaStreamWaitEvent,
cudaStreamSynchronize, cudaStreamAddCallback, cuStreamDestroy
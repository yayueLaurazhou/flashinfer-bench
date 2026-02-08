# __host__cudaError_t cudaStreamQuery (cudaStream_t stream)

Queries an asynchronous stream for completion status.

##### Parameters

**stream**

  - Stream identifier

##### Returns

cudaSuccess, cudaErrorNotReady, cudaErrorInvalidResourceHandle

##### Description

Returns cudaSuccess if all operations in stream have completed, or cudaErrorNotReady if not.

For the purposes of Unified Memory, a return value of cudaSuccess is equivalent to having called
cudaStreamSynchronize().











See also:


CUDA Runtime API vRelease Version  |  67


Modules


cudaStreamCreate, cudaStreamCreateWithFlags, cudaStreamWaitEvent, cudaStreamSynchronize,
cudaStreamAddCallback, cudaStreamDestroy, cuStreamQuery
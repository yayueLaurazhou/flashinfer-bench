# __host__cudaError_t cudaEGLStreamConsumerReleaseFrame (cudaEglStreamConnection *conn, cudaGraphicsResource_t pCudaResource, cudaStream_t *pStream)

Releases the last frame acquired from the EGLStream.

##### Parameters

**conn**

  - Connection on which to release
**pCudaResource**

  - CUDA resource whose corresponding frame is to be released
**pStream**

  - CUDA stream on which release will be done.

##### Returns

cudaSuccess, cudaErrorInvalidValue, cudaErrorUnknown

##### Description

Release the acquired image frame specified by pCudaResource to EGLStreamKHR.


See also:

cudaEGLStreamConsumerConnect, cudaEGLStreamConsumerDisconnect,
cudaEGLStreamConsumerAcquireFrame, cuEGLStreamConsumerReleaseFrame


CUDA Runtime API vRelease Version  |  289


Modules
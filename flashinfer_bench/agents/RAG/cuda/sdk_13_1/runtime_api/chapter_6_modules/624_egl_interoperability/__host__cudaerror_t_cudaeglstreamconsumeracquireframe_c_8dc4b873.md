# __host__cudaError_t cudaEGLStreamConsumerAcquireFrame (cudaEglStreamConnection *conn, cudaGraphicsResource_t *pCudaResource, cudaStream_t *pStream, unsigned int timeout)

Acquire an image frame from the EGLStream with CUDA as a consumer.

##### Parameters

**conn**

  - Connection on which to acquire
**pCudaResource**

  - CUDA resource on which the EGLStream frame will be mapped for use.
**pStream**

  - CUDA stream for synchronization and any data migrations implied by
cudaEglResourceLocationFlags.
**timeout**

  - Desired timeout in usec.

##### Returns

cudaSuccess, cudaErrorInvalidValue, cudaErrorUnknown, cudaErrorLaunchTimeout


CUDA Runtime API vRelease Version  |  286


Modules

##### Description

Acquire an image frame from EGLStreamKHR. cudaGraphicsResourceGetMappedEglFrame can be
called on pCudaResource to get cudaEglFrame.


See also:

cudaEGLStreamConsumerConnect, cudaEGLStreamConsumerDisconnect,
cudaEGLStreamConsumerReleaseFrame, cuEGLStreamConsumerAcquireFrame
# CUresult cuEGLStreamConsumerAcquireFrame (CUeglStreamConnection *conn, CUgraphicsResource *pCudaResource, CUstream *pStream, unsigned int timeout)

Acquire an image frame from the EGLStream with CUDA as a consumer.

###### Parameters

**conn**

  - Connection on which to acquire
**pCudaResource**

  - CUDA resource on which the stream frame will be mapped for use.
**pStream**

  - CUDA stream for synchronization and any data migrations implied by
CUeglResourceLocationFlags.
**timeout**

  - Desired timeout in usec for a new frame to be acquired. If set as
CUDA_EGL_INFINITE_TIMEOUT, acquire waits infinitely. After timeout occurs CUDA
consumer tries to acquire an old frame if available and EGL_SUPPORT_REUSE_NV flag is set.

###### Returns

CUDA_SUCCESS, CUDA_ERROR_INVALID_HANDLE, CUDA_ERROR_LAUNCH_TIMEOUT,

###### Description

Acquire an image frame from EGLStreamKHR. This API can also acquire an old frame presented by
the producer unless explicitly disabled by setting EGL_SUPPORT_REUSE_NV flag to EGL_FALSE
during stream initialization. By default, EGLStream is created with this flag set to EGL_TRUE.
cuGraphicsResourceGetMappedEglFrame can be called on pCudaResource to get CUeglFrame.


See also:

cuEGLStreamConsumerConnect, cuEGLStreamConsumerDisconnect,
cuEGLStreamConsumerAcquireFrame, cuEGLStreamConsumerReleaseFrame,
cudaEGLStreamConsumerAcquireFrame


CUDA Driver API TRM-06703-001 _vRelease Version  |  660


Modules
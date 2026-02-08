# CUresult cuEGLStreamConsumerReleaseFrame (CUeglStreamConnection *conn, CUgraphicsResource pCudaResource, CUstream *pStream)

Releases the last frame acquired from the EGLStream.

###### Parameters

**conn**

  - Connection on which to release
**pCudaResource**

  - CUDA resource whose corresponding frame is to be released
**pStream**

  - CUDA stream on which release will be done.

###### Returns

CUDA_SUCCESS, CUDA_ERROR_INVALID_HANDLE,

###### Description

Release the acquired image frame specified by pCudaResource to EGLStreamKHR. If
EGL_SUPPORT_REUSE_NV flag is set to EGL_TRUE, at the time of EGL creation this API doesn't
release the last frame acquired on the EGLStream. By default, EGLStream is created with this flag set
to EGL_TRUE.


See also:

cuEGLStreamConsumerConnect, cuEGLStreamConsumerDisconnect,
cuEGLStreamConsumerAcquireFrame, cuEGLStreamConsumerReleaseFrame,
cudaEGLStreamConsumerReleaseFrame
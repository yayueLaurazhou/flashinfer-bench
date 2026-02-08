# CUresult cuEGLStreamConsumerConnect (CUeglStreamConnection *conn, EGLStreamKHR stream)

Connect CUDA to EGLStream as a consumer.

###### Parameters

**conn**

  - Pointer to the returned connection handle
**stream**

  - EGLStreamKHR handle

###### Returns

CUDA_SUCCESS, CUDA_ERROR_INVALID_HANDLE, CUDA_ERROR_INVALID_CONTEXT,

###### Description

Connect CUDA as a consumer to EGLStreamKHR specified by stream.

The EGLStreamKHR is an EGL object that transfers a sequence of image frames from one API to
another.


See also:

cuEGLStreamConsumerConnect, cuEGLStreamConsumerDisconnect,
cuEGLStreamConsumerAcquireFrame, cuEGLStreamConsumerReleaseFrame,
cudaEGLStreamConsumerConnect
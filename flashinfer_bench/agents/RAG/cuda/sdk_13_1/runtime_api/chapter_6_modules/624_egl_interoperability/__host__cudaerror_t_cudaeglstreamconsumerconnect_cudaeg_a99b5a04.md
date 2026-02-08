# __host__cudaError_t cudaEGLStreamConsumerConnect (cudaEglStreamConnection *conn, EGLStreamKHR eglStream)

Connect CUDA to EGLStream as a consumer.

##### Parameters

**conn**

  - Pointer to the returned connection handle
**eglStream**

  - EGLStreamKHR handle

##### Returns

cudaSuccess, cudaErrorInvalidValue, cudaErrorUnknown

##### Description

Connect CUDA as a consumer to EGLStreamKHR specified by eglStream.

The EGLStreamKHR is an EGL object that transfers a sequence of image frames from one API to
another.


See also:

cudaEGLStreamConsumerDisconnect, cudaEGLStreamConsumerAcquireFrame,
cudaEGLStreamConsumerReleaseFrame, cuEGLStreamConsumerConnect


CUDA Runtime API vRelease Version  |  287


Modules
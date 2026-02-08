# __host__cudaError_t cudaEGLStreamProducerConnect (cudaEglStreamConnection *conn, EGLStreamKHR eglStream, EGLint width, EGLint height)

Connect CUDA to EGLStream as a producer.

##### Parameters

**conn**

  - Pointer to the returned connection handle
**eglStream**

  - EGLStreamKHR handle
**width**

  - width of the image to be submitted to the stream
**height**

  - height of the image to be submitted to the stream

##### Returns

cudaSuccess, cudaErrorInvalidValue, cudaErrorUnknown

##### Description

Connect CUDA as a producer to EGLStreamKHR specified by stream.

The EGLStreamKHR is an EGL object that transfers a sequence of image frames from one API to
another.


See also:

cudaEGLStreamProducerDisconnect, cudaEGLStreamProducerPresentFrame,
cudaEGLStreamProducerReturnFrame, cuEGLStreamProducerConnect
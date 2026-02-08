# CUresult cuEGLStreamProducerConnect (CUeglStreamConnection *conn, EGLStreamKHR stream, EGLint width, EGLint height)

Connect CUDA to EGLStream as a producer.

###### Parameters

**conn**

  - Pointer to the returned connection handle
**stream**

  - EGLStreamKHR handle


CUDA Driver API TRM-06703-001 _vRelease Version  |  663


Modules


**width**

  - width of the image to be submitted to the stream
**height**

  - height of the image to be submitted to the stream

###### Returns

CUDA_SUCCESS, CUDA_ERROR_INVALID_HANDLE, CUDA_ERROR_INVALID_CONTEXT,

###### Description

Connect CUDA as a producer to EGLStreamKHR specified by stream.

The EGLStreamKHR is an EGL object that transfers a sequence of image frames from one API to
another.


See also:

cuEGLStreamProducerConnect, cuEGLStreamProducerDisconnect,
cuEGLStreamProducerPresentFrame, cudaEGLStreamProducerConnect
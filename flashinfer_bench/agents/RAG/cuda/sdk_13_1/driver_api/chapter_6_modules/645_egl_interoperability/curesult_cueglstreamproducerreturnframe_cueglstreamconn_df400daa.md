# CUresult cuEGLStreamProducerReturnFrame (CUeglStreamConnection *conn, CUeglFrame *eglframe, CUstream *pStream)

Return the CUDA eglFrame to the EGLStream released by the consumer.

###### Parameters

**conn**

  - Connection on which to return
**eglframe**

  - CUDA Eglstream Proucer Frame handle returned from the consumer over EglStream.
**pStream**

  - CUDA stream on which to return the frame.

###### Returns

CUDA_SUCCESS, CUDA_ERROR_INVALID_HANDLE, CUDA_ERROR_LAUNCH_TIMEOUT

###### Description

This API can potentially return CUDA_ERROR_LAUNCH_TIMEOUT if the consumer has not
returned a frame to EGL stream. If timeout is returned the application can retry.


See also:

cuEGLStreamProducerConnect, cuEGLStreamProducerDisconnect,
cuEGLStreamProducerPresentFrame, cudaEGLStreamProducerReturnFrame
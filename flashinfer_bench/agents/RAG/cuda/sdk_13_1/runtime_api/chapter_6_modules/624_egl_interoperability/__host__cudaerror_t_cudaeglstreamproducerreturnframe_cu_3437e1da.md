# __host__cudaError_t cudaEGLStreamProducerReturnFrame (cudaEglStreamConnection *conn, cudaEglFrame *eglframe, cudaStream_t *pStream)

Return the CUDA eglFrame to the EGLStream last released by the consumer.

##### Parameters

**conn**

  - Connection on which to present the CUDA array
**eglframe**

  - CUDA Eglstream Proucer Frame handle returned from the consumer over EglStream.
**pStream**

  - CUDA stream on which to return the frame.

##### Returns

cudaSuccess, cudaErrorLaunchTimeout, cudaErrorInvalidValue, cudaErrorUnknown

##### Description

This API can potentially return cudaErrorLaunchTimeout if the consumer has not returned a frame to
EGL stream. If timeout is returned the application can retry.


See also:

cudaEGLStreamProducerConnect, cudaEGLStreamProducerDisconnect,
cudaEGLStreamProducerPresentFrame, cuEGLStreamProducerReturnFrame
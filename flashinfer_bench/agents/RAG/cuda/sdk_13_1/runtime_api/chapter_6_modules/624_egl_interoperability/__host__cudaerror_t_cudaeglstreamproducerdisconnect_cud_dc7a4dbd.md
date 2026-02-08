# __host__cudaError_t cudaEGLStreamProducerDisconnect (cudaEglStreamConnection *conn)

Disconnect CUDA as a producer to EGLStream .

##### Parameters

**conn**

  - Conection to disconnect.

##### Returns

cudaSuccess, cudaErrorInvalidValue, cudaErrorUnknown


CUDA Runtime API vRelease Version  |  290


Modules

##### Description

Disconnect CUDA as a producer to EGLStreamKHR.


See also:

cudaEGLStreamProducerConnect, cudaEGLStreamProducerPresentFrame,
cudaEGLStreamProducerReturnFrame, cuEGLStreamProducerDisconnect
# __host__cudaError_t cudaEGLStreamConsumerDisconnect (cudaEglStreamConnection *conn)

Disconnect CUDA as a consumer to EGLStream .

##### Parameters

**conn**

  - Conection to disconnect.

##### Returns

cudaSuccess, cudaErrorInvalidValue, cudaErrorUnknown


CUDA Runtime API vRelease Version  |  288


Modules

##### Description

Disconnect CUDA as a consumer to EGLStreamKHR.


See also:

cudaEGLStreamConsumerConnect, cudaEGLStreamConsumerAcquireFrame,
cudaEGLStreamConsumerReleaseFrame, cuEGLStreamConsumerDisconnect
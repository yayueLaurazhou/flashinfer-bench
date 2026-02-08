# CUresult cuEGLStreamProducerDisconnect (CUeglStreamConnection *conn)

Disconnect CUDA as a producer to EGLStream .

###### Parameters

**conn**

  - Conection to disconnect.

###### Returns

CUDA_SUCCESS, CUDA_ERROR_INVALID_HANDLE, CUDA_ERROR_INVALID_CONTEXT,

###### Description

Disconnect CUDA as a producer to EGLStreamKHR.


See also:

cuEGLStreamProducerConnect, cuEGLStreamProducerDisconnect,
cuEGLStreamProducerPresentFrame, cudaEGLStreamProducerDisconnect


CUDA Driver API TRM-06703-001 _vRelease Version  |  664


Modules
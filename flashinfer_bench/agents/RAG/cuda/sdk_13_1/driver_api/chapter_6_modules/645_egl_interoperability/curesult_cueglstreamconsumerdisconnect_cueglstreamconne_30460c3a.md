# CUresult cuEGLStreamConsumerDisconnect (CUeglStreamConnection *conn)

Disconnect CUDA as a consumer to EGLStream .

###### Parameters

**conn**

  - Conection to disconnect.

###### Returns

CUDA_SUCCESS, CUDA_ERROR_INVALID_HANDLE, CUDA_ERROR_INVALID_CONTEXT,

###### Description

Disconnect CUDA as a consumer to EGLStreamKHR.


See also:

cuEGLStreamConsumerConnect, cuEGLStreamConsumerDisconnect,
cuEGLStreamConsumerAcquireFrame, cuEGLStreamConsumerReleaseFrame,
cudaEGLStreamConsumerDisconnect


CUDA Driver API TRM-06703-001 _vRelease Version  |  662


Modules
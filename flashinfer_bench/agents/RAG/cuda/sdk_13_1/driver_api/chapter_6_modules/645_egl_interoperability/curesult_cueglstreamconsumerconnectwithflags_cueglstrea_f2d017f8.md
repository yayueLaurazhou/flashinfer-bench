# CUresult cuEGLStreamConsumerConnectWithFlags (CUeglStreamConnection *conn, EGLStreamKHR stream, unsigned int flags)

Connect CUDA to EGLStream as a consumer with given flags.

###### Parameters

**conn**

  - Pointer to the returned connection handle
**stream**

  - EGLStreamKHR handle
**flags**

  - Flags denote intended location - system or video.


CUDA Driver API TRM-06703-001 _vRelease Version  |  661


Modules

###### Returns

CUDA_SUCCESS, CUDA_ERROR_INVALID_HANDLE, CUDA_ERROR_INVALID_CONTEXT,

###### Description

Connect CUDA as a consumer to EGLStreamKHR specified by stream with specified flags
defined by CUeglResourceLocationFlags.

The flags specify whether the consumer wants to access frames from system memory or video memory.
Default is CU_EGL_RESOURCE_LOCATION_VIDMEM.


See also:

cuEGLStreamConsumerConnect, cuEGLStreamConsumerDisconnect,
cuEGLStreamConsumerAcquireFrame, cuEGLStreamConsumerReleaseFrame,
cudaEGLStreamConsumerConnectWithFlags
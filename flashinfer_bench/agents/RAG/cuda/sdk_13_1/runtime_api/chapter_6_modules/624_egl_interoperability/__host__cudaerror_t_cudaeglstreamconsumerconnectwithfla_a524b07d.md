# __host__cudaError_t cudaEGLStreamConsumerConnectWithFlags (cudaEglStreamConnection *conn, EGLStreamKHR eglStream, unsigned int flags)

Connect CUDA to EGLStream as a consumer with given flags.

##### Parameters

**conn**

  - Pointer to the returned connection handle
**eglStream**

  - EGLStreamKHR handle
**flags**

  - Flags denote intended location - system or video.

##### Returns

cudaSuccess, cudaErrorInvalidValue, cudaErrorUnknown

##### Description

Connect CUDA as a consumer to EGLStreamKHR specified by stream with specified flags
defined by cudaEglResourceLocationFlags.

The flags specify whether the consumer wants to access frames from system memory or video memory.
Default is cudaEglResourceLocationVidmem.


See also:

cudaEGLStreamConsumerDisconnect, cudaEGLStreamConsumerAcquireFrame,
cudaEGLStreamConsumerReleaseFrame, cuEGLStreamConsumerConnectWithFlags
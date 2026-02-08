# CUresult cuEGLStreamProducerPresentFrame (CUeglStreamConnection *conn, CUeglFrame eglframe, CUstream *pStream)

Present a CUDA eglFrame to the EGLStream with CUDA as a producer.

###### Parameters

**conn**

  - Connection on which to present the CUDA array
**eglframe**

  - CUDA Eglstream Proucer Frame handle to be sent to the consumer over EglStream.
**pStream**

  - CUDA stream on which to present the frame.

###### Returns

CUDA_SUCCESS, CUDA_ERROR_INVALID_HANDLE,

###### Description

When a frame is presented by the producer, it gets associated with the EGLStream and thus it is illegal
to free the frame before the producer is disconnected. If a frame is freed and reused it may lead to
undefined behavior.

If producer and consumer are on different GPUs (iGPU and dGPU) then frametype
CU_EGL_FRAME_TYPE_ARRAY is not supported. CU_EGL_FRAME_TYPE_PITCH can be used
for such cross-device applications.

The CUeglFrame is defined as:


For CUeglFrame of type CU_EGL_FRAME_TYPE_PITCH, the application may present sub-region of
a memory allocation. In that case, the pitched pointer will specify the start address of the sub-region in
the allocation and corresponding CUeglFrame fields will specify the dimensions of the sub-region.


See also:


CUDA Driver API TRM-06703-001 _vRelease Version  |  665


Modules


cuEGLStreamProducerConnect, cuEGLStreamProducerDisconnect,
cuEGLStreamProducerReturnFrame, cudaEGLStreamProducerPresentFrame
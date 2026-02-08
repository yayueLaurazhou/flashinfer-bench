# __host__cudaError_t cudaEGLStreamProducerPresentFrame (cudaEglStreamConnection *conn, cudaEglFrame eglframe, cudaStream_t *pStream)

Present a CUDA eglFrame to the EGLStream with CUDA as a producer.

##### Parameters

**conn**

  - Connection on which to present the CUDA array
**eglframe**

  - CUDA Eglstream Proucer Frame handle to be sent to the consumer over EglStream.
**pStream**

  - CUDA stream on which to present the frame.

##### Returns

cudaSuccess, cudaErrorInvalidValue, cudaErrorUnknown

##### Description

The cudaEglFrame is defined as:


For cudaEglFrame of type cudaEglFrameTypePitch, the application may present sub-region of a
memory allocation. In that case, cudaPitchedPtr::ptr will specify the start address of the sub-region in
the allocation and cudaEglPlaneDesc will specify the dimensions of the sub-region.


See also:


CUDA Runtime API vRelease Version  |  291


Modules


cudaEGLStreamProducerConnect, cudaEGLStreamProducerDisconnect,
cudaEGLStreamProducerReturnFrame, cuEGLStreamProducerPresentFrame
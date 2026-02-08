# __host____device__cudaError_t cudaMemset2DAsync (void *devPtr, size_t pitch, int value, size_t width, size_t height, cudaStream_t stream)

Initializes or sets device memory to a value.

##### Parameters

**devPtr**

  - Pointer to 2D device memory
**pitch**

  - Pitch in bytes of 2D device memory(Unused if height is 1)
**value**

  - Value to set for each byte of specified memory
**width**

  - Width of matrix set (columns in bytes)
**height**

  - Height of matrix set (rows)
**stream**

  - Stream identifier

##### Returns

cudaSuccess, cudaErrorInvalidValue,

##### Description

Sets to the specified value value a matrix (height rows of width bytes each) pointed to by
dstPtr. pitch is the width in bytes of the 2D array pointed to by dstPtr, including any padding
added to the end of each row. This function performs fastest when the pitch is one that has been passed
back by cudaMallocPitch().


CUDA Runtime API vRelease Version  |  195


Modules


cudaMemset2DAsync() is asynchronous with respect to the host, so the call may return before the
memset is complete. The operation can optionally be associated to a stream by passing a non-zero
stream argument. If stream is non-zero, the operation may overlap with operations in other
streams.

The device version of this function only handles device to device copies and cannot be given local or
shared pointers.













See also:

cudaMemset, cudaMemset2D, cudaMemset3D, cudaMemsetAsync, cudaMemset3DAsync,
cuMemsetD2D8Async, cuMemsetD2D16Async, cuMemsetD2D32Async
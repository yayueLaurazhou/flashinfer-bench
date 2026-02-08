# __host____device__cudaError_t cudaMemsetAsync (void *devPtr, int value, size_t count, cudaStream_t stream)

Initializes or sets device memory to a value.

##### Parameters

**devPtr**

  - Pointer to device memory
**value**

  - Value to set for each byte of specified memory
**count**

  - Size in bytes to set
**stream**

  - Stream identifier

##### Returns

cudaSuccess, cudaErrorInvalidValue,

##### Description

Fills the first count bytes of the memory area pointed to by devPtr with the constant byte value
value.

cudaMemsetAsync() is asynchronous with respect to the host, so the call may return before the memset
is complete. The operation can optionally be associated to a stream by passing a non-zero stream
argument. If stream is non-zero, the operation may overlap with operations in other streams.

The device version of this function only handles device to device copies and cannot be given local or
shared pointers.













CUDA Runtime API vRelease Version  |  199


Modules


See also:

cudaMemset, cudaMemset2D, cudaMemset3D, cudaMemset2DAsync, cudaMemset3DAsync,
cuMemsetD8Async, cuMemsetD16Async, cuMemsetD32Async
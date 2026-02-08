# __host__cudaError_t cudaMemset (void *devPtr, int value, size_t count)

Initializes or sets device memory to a value.

##### Parameters

**devPtr**

  - Pointer to device memory
**value**

  - Value to set for each byte of specified memory
**count**

  - Size in bytes to set

##### Returns

cudaSuccess, cudaErrorInvalidValue,

##### Description

Fills the first count bytes of the memory area pointed to by devPtr with the constant byte value
value.

Note that this function is asynchronous with respect to the host unless devPtr refers to pinned host
memory.


Note:


CUDA Runtime API vRelease Version  |  193


Modules









See also:

cuMemsetD8, cuMemsetD16, cuMemsetD32
# __host__cudaError_t cudaMemset2D (void *devPtr, size_t pitch, int value, size_t width, size_t height)

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

##### Returns

cudaSuccess, cudaErrorInvalidValue,

##### Description

Sets to the specified value value a matrix (height rows of width bytes each) pointed to by
dstPtr. pitch is the width in bytes of the 2D array pointed to by dstPtr, including any padding
added to the end of each row. This function performs fastest when the pitch is one that has been passed
back by cudaMallocPitch().

Note that this function is asynchronous with respect to the host unless devPtr refers to pinned host
memory.


Note:


CUDA Runtime API vRelease Version  |  194


Modules









See also:

cudaMemset, cudaMemset3D, cudaMemsetAsync, cudaMemset2DAsync, cudaMemset3DAsync,
cuMemsetD2D8, cuMemsetD2D16, cuMemsetD2D32
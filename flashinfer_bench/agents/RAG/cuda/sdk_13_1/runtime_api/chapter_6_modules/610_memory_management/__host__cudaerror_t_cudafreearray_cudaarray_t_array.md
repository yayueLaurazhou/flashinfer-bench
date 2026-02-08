# __host__cudaError_t cudaFreeArray (cudaArray_t array)

Frees an array on the device.

##### Parameters

**array**

  - Pointer to array to free

##### Returns

cudaSuccess, cudaErrorInvalidValue


CUDA Runtime API vRelease Version  |  122


Modules

##### Description

Frees the CUDA array array, which must have been returned by a previous call to
cudaMallocArray(). If devPtr is 0, no operation is performed.





See also:

cudaMalloc, cudaMallocPitch, cudaFree, cudaMallocArray, cudaMallocHost ( C API), cudaFreeHost,
cudaHostAlloc, cuArrayDestroy
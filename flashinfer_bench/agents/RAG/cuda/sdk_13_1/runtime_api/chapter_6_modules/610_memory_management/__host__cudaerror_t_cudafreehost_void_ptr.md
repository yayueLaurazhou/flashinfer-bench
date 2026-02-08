# __host__cudaError_t cudaFreeHost (void *ptr)

Frees page-locked memory.

##### Parameters

**ptr**

  - Pointer to memory to free

##### Returns

cudaSuccess, cudaErrorInvalidValue

##### Description

Frees the memory space pointed to by hostPtr, which must have been returned by a previous call to
cudaMallocHost() or cudaHostAlloc().





CUDA Runtime API vRelease Version  |  123


Modules


See also:

cudaMalloc, cudaMallocPitch, cudaFree, cudaMallocArray, cudaFreeArray, cudaMallocHost ( C API),
cudaMalloc3D, cudaMalloc3DArray, cudaHostAlloc, cuMemFreeHost
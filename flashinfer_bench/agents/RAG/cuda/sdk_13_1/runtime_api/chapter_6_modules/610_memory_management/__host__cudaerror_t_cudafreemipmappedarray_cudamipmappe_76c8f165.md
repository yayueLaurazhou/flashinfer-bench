# __host__cudaError_t cudaFreeMipmappedArray (cudaMipmappedArray_t mipmappedArray)

Frees a mipmapped array on the device.

##### Parameters

**mipmappedArray**

  - Pointer to mipmapped array to free

##### Returns

cudaSuccess, cudaErrorInvalidValue

##### Description

Frees the CUDA mipmapped array mipmappedArray, which must have been returned by a previous
call to cudaMallocMipmappedArray(). If devPtr is 0, no operation is performed.



See also:

cudaMalloc, cudaMallocPitch, cudaFree, cudaMallocArray, cudaMallocHost ( C API), cudaFreeHost,
cudaHostAlloc, cuMipmappedArrayDestroy


CUDA Runtime API vRelease Version  |  124


Modules
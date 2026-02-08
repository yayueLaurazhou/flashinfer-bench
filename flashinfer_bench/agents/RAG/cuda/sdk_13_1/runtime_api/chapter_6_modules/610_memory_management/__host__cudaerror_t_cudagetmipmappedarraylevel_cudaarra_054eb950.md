# __host__cudaError_t cudaGetMipmappedArrayLevel (cudaArray_t *levelArray, cudaMipmappedArray_const_t mipmappedArray, unsigned int level)

Gets a mipmap level of a CUDA mipmapped array.

##### Parameters

**levelArray**

  - Returned mipmap level CUDA array
**mipmappedArray**

  - CUDA mipmapped array
**level**

  - Mipmap level

##### Returns

cudaSuccess, cudaErrorInvalidValue cudaErrorInvalidResourceHandle

##### Description

Returns in *levelArray a CUDA array that represents a single mipmap level of the CUDA
mipmapped array mipmappedArray.

If level is greater than the maximum number of levels in this mipmapped array,
cudaErrorInvalidValue is returned.

If mipmappedArray is NULL, cudaErrorInvalidResourceHandle is returned.



See also:

cudaMalloc3D, cudaMalloc, cudaMallocPitch, cudaFree, cudaFreeArray, cudaMallocHost ( C API),
cudaFreeHost, cudaHostAlloc, make_cudaExtent, cuMipmappedArrayGetLevel


CUDA Runtime API vRelease Version  |  125


Modules
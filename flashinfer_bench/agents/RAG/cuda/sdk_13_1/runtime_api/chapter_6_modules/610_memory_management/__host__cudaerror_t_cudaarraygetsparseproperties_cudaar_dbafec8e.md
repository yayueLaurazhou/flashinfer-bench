# __host__cudaError_t cudaArrayGetSparseProperties (cudaArraySparseProperties *sparseProperties, cudaArray_t array)

Returns the layout properties of a sparse CUDA array.

##### Parameters

**sparseProperties**

  - Pointer to return the cudaArraySparseProperties
**array**

  - The CUDA array to get the sparse properties of

##### Returns

cudaSuccess cudaErrorInvalidValue

##### Description

Returns the layout properties of a sparse CUDA array in sparseProperties. If the CUDA array is
not allocated with flag cudaArraySparse cudaErrorInvalidValue will be returned.

If the returned value in cudaArraySparseProperties::flags contains
cudaArraySparsePropertiesSingleMipTail, then cudaArraySparseProperties::miptailSize
represents the total size of the array. Otherwise, it will be zero. Also, the returned value in
cudaArraySparseProperties::miptailFirstLevel is always zero. Note that the array must have
been allocated using cudaMallocArray or cudaMalloc3DArray. For CUDA arrays obtained
using cudaMipmappedArrayGetLevel, cudaErrorInvalidValue will be returned. Instead,
cudaMipmappedArrayGetSparseProperties must be used to obtain the sparse properties of the entire
CUDA mipmapped array to which array belongs to.


See also:

cudaMipmappedArrayGetSparseProperties, cuMemMapArrayAsync
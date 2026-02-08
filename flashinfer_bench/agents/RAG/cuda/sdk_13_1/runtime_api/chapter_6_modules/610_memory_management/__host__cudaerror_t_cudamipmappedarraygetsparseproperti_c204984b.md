# __host__cudaError_t cudaMipmappedArrayGetSparseProperties (cudaArraySparseProperties *sparseProperties, cudaMipmappedArray_t mipmap)

Returns the layout properties of a sparse CUDA mipmapped array.

##### Parameters

**sparseProperties**

  - Pointer to return cudaArraySparseProperties
**mipmap**

  - The CUDA mipmapped array to get the sparse properties of

##### Returns

cudaSuccess cudaErrorInvalidValue

##### Description

Returns the sparse array layout properties in sparseProperties. If the CUDA mipmapped array is
not allocated with flag cudaArraySparse cudaErrorInvalidValue will be returned.

For non-layered CUDA mipmapped arrays, cudaArraySparseProperties::miptailSize returns the size of
the mip tail region. The mip tail region includes all mip levels whose width, height or depth is less than
that of the tile. For layered CUDA mipmapped arrays, if cudaArraySparseProperties::flags contains
cudaArraySparsePropertiesSingleMipTail, then cudaArraySparseProperties::miptailSize specifies the
size of the mip tail of all layers combined. Otherwise, cudaArraySparseProperties::miptailSize specifies
mip tail size per layer. The returned value of cudaArraySparseProperties::miptailFirstLevel is valid
only if cudaArraySparseProperties::miptailSize is non-zero.


See also:

cudaArrayGetSparseProperties, cuMemMapArrayAsync
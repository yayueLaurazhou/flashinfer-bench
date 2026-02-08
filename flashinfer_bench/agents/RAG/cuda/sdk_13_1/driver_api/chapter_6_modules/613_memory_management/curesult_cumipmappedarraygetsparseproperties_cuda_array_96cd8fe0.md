# CUresult cuMipmappedArrayGetSparseProperties (CUDA_ARRAY_SPARSE_PROPERTIES *sparseProperties, CUmipmappedArray mipmap)

Returns the layout properties of a sparse CUDA mipmapped array.

###### Parameters

**sparseProperties**

 - Pointer to CUDA_ARRAY_SPARSE_PROPERTIES
**mipmap**

  - CUDA mipmapped array to get the sparse properties of

###### Returns

CUDA_SUCCESS CUDA_ERROR_INVALID_VALUE

###### Description

Returns the sparse array layout properties in sparseProperties If the CUDA mipmapped array
is not allocated with flag CUDA_ARRAY3D_SPARSE CUDA_ERROR_INVALID_VALUE will be
returned.

For non-layered CUDA mipmapped arrays, CUDA_ARRAY_SPARSE_PROPERTIES::miptailSize
returns the size of the mip tail region. The mip tail region includes all mip
levels whose width, height or depth is less than that of the tile. For layered
CUDA mipmapped arrays, if CUDA_ARRAY_SPARSE_PROPERTIES::flags
contains CU_ARRAY_SPARSE_PROPERTIES_SINGLE_MIPTAIL, then
CUDA_ARRAY_SPARSE_PROPERTIES::miptailSize specifies the size of the mip tail of all layers
combined. Otherwise, CUDA_ARRAY_SPARSE_PROPERTIES::miptailSize specifies mip tail size


CUDA Driver API TRM-06703-001 _vRelease Version  |  270


Modules


per layer. The returned value of CUDA_ARRAY_SPARSE_PROPERTIES::miptailFirstLevel is valid
only if CUDA_ARRAY_SPARSE_PROPERTIES::miptailSize is non-zero.


See also:

cuArrayGetSparseProperties, cuMemMapArrayAsync
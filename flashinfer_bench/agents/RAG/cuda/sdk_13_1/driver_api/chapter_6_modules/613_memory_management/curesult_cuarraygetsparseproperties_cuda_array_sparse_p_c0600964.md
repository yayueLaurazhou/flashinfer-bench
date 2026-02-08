# CUresult cuArrayGetSparseProperties (CUDA_ARRAY_SPARSE_PROPERTIES *sparseProperties, CUarray array)

Returns the layout properties of a sparse CUDA array.

###### Parameters

**sparseProperties**

 - Pointer to CUDA_ARRAY_SPARSE_PROPERTIES
**array**

  - CUDA array to get the sparse properties of


CUDA Driver API TRM-06703-001 _vRelease Version  |  186


Modules

###### Returns

CUDA_SUCCESS CUDA_ERROR_INVALID_VALUE

###### Description

Returns the layout properties of a sparse CUDA array in sparseProperties If the CUDA array
is not allocated with flag CUDA_ARRAY3D_SPARSE CUDA_ERROR_INVALID_VALUE will be
returned.

If the returned value in CUDA_ARRAY_SPARSE_PROPERTIES::flags
contains CU_ARRAY_SPARSE_PROPERTIES_SINGLE_MIPTAIL,
then CUDA_ARRAY_SPARSE_PROPERTIES::miptailSize represents the
total size of the array. Otherwise, it will be zero. Also, the returned value in
CUDA_ARRAY_SPARSE_PROPERTIES::miptailFirstLevel is always zero. Note that the array
must have been allocated using cuArrayCreate or cuArray3DCreate. For CUDA arrays obtained
using cuMipmappedArrayGetLevel, CUDA_ERROR_INVALID_VALUE will be returned. Instead,
cuMipmappedArrayGetSparseProperties must be used to obtain the sparse properties of the entire
CUDA mipmapped array to which array belongs to.


See also:

cuMipmappedArrayGetSparseProperties, cuMemMapArrayAsync
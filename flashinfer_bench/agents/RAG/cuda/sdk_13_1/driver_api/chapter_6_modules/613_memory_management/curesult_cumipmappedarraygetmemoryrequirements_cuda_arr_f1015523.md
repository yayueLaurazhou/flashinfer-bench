# CUresult cuMipmappedArrayGetMemoryRequirements (CUDA_ARRAY_MEMORY_REQUIREMENTS *memoryRequirements, CUmipmappedArray mipmap, CUdevice device)

Returns the memory requirements of a CUDA mipmapped array.

###### Parameters

**memoryRequirements**

 - Pointer to CUDA_ARRAY_MEMORY_REQUIREMENTS
**mipmap**

  - CUDA mipmapped array to get the memory requirements of
**device**

  - Device to get the memory requirements for

###### Returns

CUDA_SUCCESS CUDA_ERROR_INVALID_VALUE


CUDA Driver API TRM-06703-001 _vRelease Version  |  269


Modules

###### Description

Returns the memory requirements of a CUDA mipmapped array in memoryRequirements If the
CUDA mipmapped array is not allocated with flag CUDA_ARRAY3D_DEFERRED_MAPPING
CUDA_ERROR_INVALID_VALUE will be returned.

The returned value in CUDA_ARRAY_MEMORY_REQUIREMENTS::size
represents the total size of the CUDA mipmapped array. The returned value in
CUDA_ARRAY_MEMORY_REQUIREMENTS::alignment represents the alignment necessary for
mapping the CUDA mipmapped array.


See also:

cuArrayGetMemoryRequirements, cuMemMapArrayAsync
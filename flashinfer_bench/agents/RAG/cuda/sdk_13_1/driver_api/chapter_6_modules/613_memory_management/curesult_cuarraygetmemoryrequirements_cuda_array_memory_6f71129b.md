# CUresult cuArrayGetMemoryRequirements (CUDA_ARRAY_MEMORY_REQUIREMENTS *memoryRequirements, CUarray array, CUdevice device)

Returns the memory requirements of a CUDA array.

###### Parameters

**memoryRequirements**

 - Pointer to CUDA_ARRAY_MEMORY_REQUIREMENTS
**array**

  - CUDA array to get the memory requirements of
**device**

  - Device to get the memory requirements for

###### Returns

CUDA_SUCCESS CUDA_ERROR_INVALID_VALUE

###### Description

Returns the memory requirements of a CUDA array in memoryRequirements If the
CUDA array is not allocated with flag CUDA_ARRAY3D_DEFERRED_MAPPING
CUDA_ERROR_INVALID_VALUE will be returned.

The returned value in CUDA_ARRAY_MEMORY_REQUIREMENTS::size
represents the total size of the CUDA array. The returned value in
CUDA_ARRAY_MEMORY_REQUIREMENTS::alignment represents the alignment necessary for
mapping the CUDA array.


See also:

cuMipmappedArrayGetMemoryRequirements, cuMemMapArrayAsync
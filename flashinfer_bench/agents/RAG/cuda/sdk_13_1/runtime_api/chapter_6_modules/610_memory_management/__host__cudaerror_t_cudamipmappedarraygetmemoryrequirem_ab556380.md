# __host__cudaError_t cudaMipmappedArrayGetMemoryRequirements (cudaArrayMemoryRequirements *memoryRequirements, cudaMipmappedArray_t mipmap, int device)

Returns the memory requirements of a CUDA mipmapped array.

##### Parameters

**memoryRequirements**

  - Pointer to cudaArrayMemoryRequirements
**mipmap**

  - CUDA mipmapped array to get the memory requirements of
**device**

  - Device to get the memory requirements for

##### Returns

cudaSuccess cudaErrorInvalidValue

##### Description

Returns the memory requirements of a CUDA mipmapped array in memoryRequirements If the
CUDA mipmapped array is not allocated with flag cudaArrayDeferredMapping cudaErrorInvalidValue
will be returned.

The returned value in cudaArrayMemoryRequirements::size represents the total size of the CUDA
mipmapped array. The returned value in cudaArrayMemoryRequirements::alignment represents the
alignment necessary for mapping the CUDA mipmapped array.


See also:

cudaArrayGetMemoryRequirements


CUDA Runtime API vRelease Version  |  200


Modules
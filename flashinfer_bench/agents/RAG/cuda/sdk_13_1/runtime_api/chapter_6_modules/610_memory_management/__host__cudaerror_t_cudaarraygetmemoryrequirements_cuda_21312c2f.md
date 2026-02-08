# __host__cudaError_t cudaArrayGetMemoryRequirements (cudaArrayMemoryRequirements *memoryRequirements, cudaArray_t array, int device)

Returns the memory requirements of a CUDA array.

##### Parameters

**memoryRequirements**

  - Pointer to cudaArrayMemoryRequirements
**array**

  - CUDA array to get the memory requirements of
**device**

  - Device to get the memory requirements for

##### Returns

cudaSuccess cudaErrorInvalidValue

##### Description

Returns the memory requirements of a CUDA array in memoryRequirements If the CUDA array
is not allocated with flag cudaArrayDeferredMapping cudaErrorInvalidValue will be returned.

The returned value in cudaArrayMemoryRequirements::size represents the total size of the CUDA
array. The returned value in cudaArrayMemoryRequirements::alignment represents the alignment
necessary for mapping the CUDA array.


See also:

cudaMipmappedArrayGetMemoryRequirements


CUDA Runtime API vRelease Version  |  119


Modules
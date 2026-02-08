# __host__cudaError_t cudaDestroyExternalMemory (cudaExternalMemory_t extMem)

Destroys an external memory object.

##### Parameters

**extMem**

  - External memory object to be destroyed

##### Returns

cudaSuccess, cudaErrorInvalidResourceHandle

##### Description

Destroys the specified external memory object. Any existing buffers and CUDA mipmapped arrays
mapped onto this object must no longer be used and must be explicitly freed using cudaFree and
cudaFreeMipmappedArray respectively.









See also:

cudaImportExternalMemory, cudaExternalMemoryGetMappedBuffer,
cudaExternalMemoryGetMappedMipmappedArray


CUDA Runtime API vRelease Version  |  81


Modules
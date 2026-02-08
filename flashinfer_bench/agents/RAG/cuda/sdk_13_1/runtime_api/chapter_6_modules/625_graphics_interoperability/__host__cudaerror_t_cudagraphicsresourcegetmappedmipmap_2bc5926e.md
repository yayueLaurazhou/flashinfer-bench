# __host__cudaError_t cudaGraphicsResourceGetMappedMipmappedArray (cudaMipmappedArray_t *mipmappedArray, cudaGraphicsResource_t resource)

Get a mipmapped array through which to access a mapped graphics resource.

##### Parameters

**mipmappedArray**

  - Returned mipmapped array through which resource may be accessed
**resource**

  - Mapped resource to access

##### Returns

cudaSuccess, cudaErrorInvalidValue, cudaErrorInvalidResourceHandle, cudaErrorUnknown

##### Description

Returns in *mipmappedArray a mipmapped array through which the mapped graphics resource
resource may be accessed. The value set in mipmappedArray may change every time that
resource is mapped.

If resource is not a texture then it cannot be accessed via an array and cudaErrorUnknown is
returned. If resource is not mapped then cudaErrorUnknown is returned.



See also:

cudaGraphicsResourceGetMappedPointer, cuGraphicsResourceGetMappedMipmappedArray


CUDA Runtime API vRelease Version  |  297


Modules
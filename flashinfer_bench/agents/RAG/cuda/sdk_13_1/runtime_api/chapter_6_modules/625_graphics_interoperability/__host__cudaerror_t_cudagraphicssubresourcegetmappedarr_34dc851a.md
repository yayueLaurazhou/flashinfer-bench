# __host__cudaError_t cudaGraphicsSubResourceGetMappedArray (cudaArray_t *array, cudaGraphicsResource_t resource, unsigned int arrayIndex, unsigned int mipLevel)

Get an array through which to access a subresource of a mapped graphics resource.

##### Parameters

**array**

  - Returned array through which a subresource of resource may be accessed
**resource**

  - Mapped resource to access
**arrayIndex**

  - Array index for array textures or cubemap face index as defined by cudaGraphicsCubeFace for
cubemap textures for the subresource to access
**mipLevel**

  - Mipmap level for the subresource to access

##### Returns

cudaSuccess, cudaErrorInvalidValue, cudaErrorInvalidResourceHandle, cudaErrorUnknown

##### Description

Returns in *array an array through which the subresource of the mapped graphics resource
resource which corresponds to array index arrayIndex and mipmap level mipLevel may be
accessed. The value set in array may change every time that resource is mapped.

If resource is not a texture then it cannot be accessed via an array and cudaErrorUnknown is
returned. If arrayIndex is not a valid array index for resource then cudaErrorInvalidValue is
returned. If mipLevel is not a valid mipmap level for resource then cudaErrorInvalidValue is
returned. If resource is not mapped then cudaErrorUnknown is returned.



See also:


CUDA Runtime API vRelease Version  |  300


Modules


cudaGraphicsResourceGetMappedPointer, cuGraphicsSubResourceGetMappedArray
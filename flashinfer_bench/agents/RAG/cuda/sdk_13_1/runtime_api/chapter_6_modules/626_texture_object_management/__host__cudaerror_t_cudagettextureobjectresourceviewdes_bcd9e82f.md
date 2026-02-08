# __host__cudaError_t cudaGetTextureObjectResourceViewDesc (cudaResourceViewDesc *pResViewDesc, cudaTextureObject_t texObject)

Returns a texture object's resource view descriptor.

##### Parameters

**pResViewDesc**

  - Resource view descriptor
**texObject**

  - Texture object

##### Returns

cudaSuccess, cudaErrorInvalidValue

##### Description

Returns the resource view descriptor for the texture object specified by texObject. If no resource
view was specified, cudaErrorInvalidValue is returned.





See also:

cudaCreateTextureObject, cuTexObjectGetResourceViewDesc


CUDA Runtime API vRelease Version  |  311


Modules
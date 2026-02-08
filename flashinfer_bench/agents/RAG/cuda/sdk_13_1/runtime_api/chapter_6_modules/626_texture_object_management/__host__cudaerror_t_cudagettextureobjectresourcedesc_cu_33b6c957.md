# __host__cudaError_t cudaGetTextureObjectResourceDesc (cudaResourceDesc *pResDesc, cudaTextureObject_t texObject)

Returns a texture object's resource descriptor.

##### Parameters

**pResDesc**

  - Resource descriptor
**texObject**

  - Texture object

##### Returns

cudaSuccess, cudaErrorInvalidValue

##### Description

Returns the resource descriptor for the texture object specified by texObject.


CUDA Runtime API vRelease Version  |  310


Modules


See also:

cudaCreateTextureObject, cuTexObjectGetResourceDesc
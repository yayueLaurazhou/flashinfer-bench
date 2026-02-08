# __host__cudaError_t cudaGetTextureObjectTextureDesc (cudaTextureDesc *pTexDesc, cudaTextureObject_t texObject)

Returns a texture object's texture descriptor.

##### Parameters

**pTexDesc**

  - Texture descriptor
**texObject**

  - Texture object

##### Returns

cudaSuccess, cudaErrorInvalidValue

##### Description

Returns the texture descriptor for the texture object specified by texObject.





See also:

cudaCreateTextureObject, cuTexObjectGetTextureDesc
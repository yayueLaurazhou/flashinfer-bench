# CUresult cuTexObjectGetTextureDesc (CUDA_TEXTURE_DESC *pTexDesc, CUtexObject texObject)

Returns a texture object's texture descriptor.

###### Parameters

**pTexDesc**

  - Texture descriptor
**texObject**

  - Texture object

###### Returns

CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE

###### Description

Returns the texture descriptor for the texture object specified by texObject.


See also:

cuTexObjectCreate, cudaGetTextureObjectTextureDesc
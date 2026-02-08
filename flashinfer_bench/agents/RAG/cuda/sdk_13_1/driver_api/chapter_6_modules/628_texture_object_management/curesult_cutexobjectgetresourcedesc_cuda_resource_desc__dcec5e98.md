# CUresult cuTexObjectGetResourceDesc (CUDA_RESOURCE_DESC *pResDesc, CUtexObject texObject)

Returns a texture object's resource descriptor.

###### Parameters

**pResDesc**

  - Resource descriptor
**texObject**

  - Texture object

###### Returns

CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE

###### Description

Returns the resource descriptor for the texture object specified by texObject.


See also:

cuTexObjectCreate, cudaGetTextureObjectResourceDesc,
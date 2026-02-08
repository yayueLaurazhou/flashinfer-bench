# CUresult cuTexObjectGetResourceViewDesc (CUDA_RESOURCE_VIEW_DESC *pResViewDesc, CUtexObject texObject)

Returns a texture object's resource view descriptor.

###### Parameters

**pResViewDesc**

  - Resource view descriptor
**texObject**

  - Texture object

###### Returns

CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE


CUDA Driver API TRM-06703-001 _vRelease Version  |  532


Modules

###### Description

Returns the resource view descriptor for the texture object specified by texObject. If no resource
view was set for texObject, the CUDA_ERROR_INVALID_VALUE is returned.


See also:

cuTexObjectCreate, cudaGetTextureObjectResourceViewDesc
# CUresult cuTexObjectDestroy (CUtexObject texObject)

Destroys a texture object.

###### Parameters

**texObject**

  - Texture object to destroy

###### Returns

CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE

###### Description

Destroys the texture object specified by texObject.


See also:

cuTexObjectCreate, cudaDestroyTextureObject


CUDA Driver API TRM-06703-001 _vRelease Version  |  531


Modules
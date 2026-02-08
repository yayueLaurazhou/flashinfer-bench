# CUresult cuSurfObjectDestroy (CUsurfObject surfObject)

Destroys a surface object.

###### Parameters

**surfObject**

  - Surface object to destroy

###### Returns

CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE

###### Description

Destroys the surface object specified by surfObject.


CUDA Driver API TRM-06703-001 _vRelease Version  |  534


Modules


See also:

cuSurfObjectCreate, cudaDestroySurfaceObject
# CUresult cuSurfRefGetArray (CUarray *phArray, CUsurfref hSurfRef)

Passes back the CUDA array bound to a surface reference.

###### Parameters

**phArray**

  - Surface reference handle
**hSurfRef**

  - Surface reference handle

###### Returns

CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE

###### Description

Deprecated

Returns in *phArray the CUDA array bound to the surface reference hSurfRef, or returns
CUDA_ERROR_INVALID_VALUE if the surface reference is not bound to any CUDA array.


CUDA Driver API TRM-06703-001 _vRelease Version  |  525


Modules


See also:

cuModuleGetSurfRef, cuSurfRefSetArray
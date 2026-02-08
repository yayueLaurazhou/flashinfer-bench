# CUresult cuSurfRefSetArray (CUsurfref hSurfRef, CUarray hArray, unsigned int Flags)

Sets the CUDA array for a surface reference.

###### Parameters

**hSurfRef**

  - Surface reference handle
**hArray**

  - CUDA array handle
**Flags**

  - set to 0

###### Returns

CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE

###### Description

Deprecated

Sets the CUDA array hArray to be read and written by the surface reference hSurfRef. Any
previous CUDA array state associated with the surface reference is superseded by this function. Flags
must be set to 0. The CUDA_ARRAY3D_SURFACE_LDST flag must have been set for the CUDA
array. Any CUDA array previously bound to hSurfRef is unbound.


See also:

cuModuleGetSurfRef, cuSurfRefGetArray
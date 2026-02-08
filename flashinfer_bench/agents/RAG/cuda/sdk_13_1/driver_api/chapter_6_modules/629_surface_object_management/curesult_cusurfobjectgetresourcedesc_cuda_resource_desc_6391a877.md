# CUresult cuSurfObjectGetResourceDesc (CUDA_RESOURCE_DESC *pResDesc, CUsurfObject surfObject)

Returns a surface object's resource descriptor.

###### Parameters

**pResDesc**

  - Resource descriptor
**surfObject**

  - Surface object

###### Returns

CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE

###### Description

Returns the resource descriptor for the surface object specified by surfObject.


See also:

cuSurfObjectCreate, cudaGetSurfaceObjectResourceDesc
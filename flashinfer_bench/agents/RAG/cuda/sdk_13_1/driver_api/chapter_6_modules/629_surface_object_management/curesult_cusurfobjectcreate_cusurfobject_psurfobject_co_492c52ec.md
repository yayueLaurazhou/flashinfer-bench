# CUresult cuSurfObjectCreate (CUsurfObject *pSurfObject, const CUDA_RESOURCE_DESC *pResDesc)

Creates a surface object.

###### Parameters

**pSurfObject**

  - Surface object to create
**pResDesc**

  - Resource descriptor

###### Returns

CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE

###### Description

Creates a surface object and returns it in pSurfObject. pResDesc describes the
data to perform surface load/stores on. CUDA_RESOURCE_DESC::resType must be
CU_RESOURCE_TYPE_ARRAY and CUDA_RESOURCE_DESC::res::array::hArray must be set to
a valid CUDA array handle. CUDA_RESOURCE_DESC::flags must be set to zero.

Surface objects are only supported on devices of compute capability 3.0 or higher. Additionally, a
surface object is an opaque value, and, as such, should only be accessed through CUDA API calls.


See also:

cuSurfObjectDestroy, cudaCreateSurfaceObject
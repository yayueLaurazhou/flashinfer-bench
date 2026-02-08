# CUresult cuModuleGetSurfRef (CUsurfref *pSurfRef, CUmodule hmod, const char *name)

Returns a handle to a surface reference.

###### Parameters

**pSurfRef**

  - Returned surface reference
**hmod**

  - Module to retrieve surface reference from
**name**

  - Name of surface reference to retrieve

###### Returns

CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE,
CUDA_ERROR_NOT_FOUND

###### Description

Deprecated

Returns in *pSurfRef the handle of the surface reference of name name in the module hmod. If no
surface reference of that name exists, cuModuleGetSurfRef() returns CUDA_ERROR_NOT_FOUND.


Note:


Note that this function may also return error codes from previous, asynchronous launches.


See also:

cuModuleGetFunction, cuModuleGetGlobal, cuModuleGetTexRef, cuModuleLoad,
cuModuleLoadData, cuModuleLoadDataEx, cuModuleLoadFatBinary, cuModuleUnload


CUDA Driver API TRM-06703-001 _vRelease Version  |  157


Modules
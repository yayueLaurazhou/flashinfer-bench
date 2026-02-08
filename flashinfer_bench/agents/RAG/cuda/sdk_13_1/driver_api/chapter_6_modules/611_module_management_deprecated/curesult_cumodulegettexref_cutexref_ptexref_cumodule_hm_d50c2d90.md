# CUresult cuModuleGetTexRef (CUtexref *pTexRef, CUmodule hmod, const char *name)

Returns a handle to a texture reference.

###### Parameters

**pTexRef**

  - Returned texture reference
**hmod**

  - Module to retrieve texture reference from
**name**

  - Name of texture reference to retrieve

###### Returns

CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE,
CUDA_ERROR_NOT_FOUND

###### Description

Deprecated

Returns in *pTexRef the handle of the texture reference of name name in the module hmod. If no
texture reference of that name exists, cuModuleGetTexRef() returns CUDA_ERROR_NOT_FOUND.
This texture reference handle should not be destroyed, since it will be destroyed when the module is
unloaded.


Note:


Note that this function may also return error codes from previous, asynchronous launches.


See also:

cuModuleGetFunction, cuModuleGetGlobal, cuModuleGetSurfRef, cuModuleLoad,
cuModuleLoadData, cuModuleLoadDataEx, cuModuleLoadFatBinary, cuModuleUnload
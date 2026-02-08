# CUresult cuModuleGetGlobal (CUdeviceptr *dptr, size_t *bytes, CUmodule hmod, const char *name)

Returns a global pointer from a module.

###### Parameters

**dptr**

  - Returned global device pointer
**bytes**

  - Returned global size in bytes
**hmod**

  - Module to retrieve global from
**name**

  - Name of global to retrieve

###### Returns

CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE,
CUDA_ERROR_NOT_FOUND

###### Description

Returns in *dptr and *bytes the base pointer and size of the global of name name
located in module hmod. If no variable of that name exists, cuModuleGetGlobal() returns
CUDA_ERROR_NOT_FOUND. One of the parameters dptr or bytes (not both) can be NULL in
which case it is ignored.


Note:


Note that this function may also return error codes from previous, asynchronous launches.


See also:


CUDA Driver API TRM-06703-001 _vRelease Version  |  151


Modules


cuModuleGetFunction, cuModuleGetTexRef, cuModuleLoad, cuModuleLoadData,
cuModuleLoadDataEx, cuModuleLoadFatBinary, cuModuleUnload, cudaGetSymbolAddress,
cudaGetSymbolSize
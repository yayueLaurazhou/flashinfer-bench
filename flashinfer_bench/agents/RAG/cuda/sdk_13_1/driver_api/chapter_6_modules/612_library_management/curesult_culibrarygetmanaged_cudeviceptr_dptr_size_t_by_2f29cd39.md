# CUresult cuLibraryGetManaged (CUdeviceptr *dptr, size_t *bytes, CUlibrary library, const char *name)

Returns a pointer to managed memory.

###### Parameters

**dptr**

  - Returned pointer to the managed memory
**bytes**

  - Returned memory size in bytes
**library**

  - Library to retrieve managed memory from
**name**

  - Name of managed memory to retrieve

###### Returns

CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
CUDA_ERROR_INVALID_VALUE, CUDA_ERROR_INVALID_HANDLE,
CUDA_ERROR_NOT_FOUND

###### Description

Returns in *dptr and *bytes the base pointer and size of the managed memory with name name
for the requested library library. If no managed memory with the requested name name exists, the
call returns CUDA_ERROR_NOT_FOUND. One of the parameters dptr or bytes (not both) can be
NULL in which case it is ignored. Note that managed memory for library library is shared across
devices and is registered when the library is loaded into atleast one context.


See also:

cuLibraryLoadData, cuLibraryLoadFromFile, cuLibraryUnload


CUDA Driver API TRM-06703-001 _vRelease Version  |  169


Modules
# CUresult cuLibraryGetGlobal (CUdeviceptr *dptr, size_t *bytes, CUlibrary library, const char *name)

Returns a global device pointer.

###### Parameters

**dptr**

  - Returned global device pointer for the requested context
**bytes**

  - Returned global size in bytes
**library**

  - Library to retrieve global from
**name**

  - Name of global to retrieve

###### Returns

CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
CUDA_ERROR_INVALID_VALUE, CUDA_ERROR_INVALID_HANDLE,
CUDA_ERROR_NOT_FOUND, CUDA_ERROR_INVALID_CONTEXT,
CUDA_ERROR_CONTEXT_IS_DESTROYED

###### Description

Returns in *dptr and *bytes the base pointer and size of the global with name name for the
requested library library and the current context. If no global for the requested name name exists,
the call returns CUDA_ERROR_NOT_FOUND. One of the parameters dptr or bytes (not both) can
be NULL in which case it is ignored.


See also:

cuLibraryLoadData, cuLibraryLoadFromFile, cuLibraryUnload, cuLibraryGetModule,
cuModuleGetGlobal


CUDA Driver API TRM-06703-001 _vRelease Version  |  167


Modules
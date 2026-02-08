# CUresult cuLibraryGetModule (CUmodule *pMod, CUlibrary library)

Returns a module handle.

###### Parameters

**pMod**

  - Returned module handle
**library**

  - Library to retrieve module from

###### Returns

CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
CUDA_ERROR_INVALID_VALUE, CUDA_ERROR_INVALID_HANDLE,
CUDA_ERROR_NOT_FOUND, CUDA_ERROR_INVALID_CONTEXT,
CUDA_ERROR_CONTEXT_IS_DESTROYED

###### Description

Returns in pMod the module handle associated with the current context located in library library. If
module handle is not found, the call returns CUDA_ERROR_NOT_FOUND.


See also:

cuLibraryLoadData, cuLibraryLoadFromFile, cuLibraryUnload, cuModuleGetFunction
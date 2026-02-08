# CUresult cuLibraryGetUnifiedFunction (void **fptr, CUlibrary library, const char *symbol)

Returns a pointer to a unified function.

###### Parameters

**fptr**

  - Returned pointer to a unified function
**library**

  - Library to retrieve function pointer memory from
**symbol**

  - Name of function pointer to retrieve


CUDA Driver API TRM-06703-001 _vRelease Version  |  170


Modules

###### Returns

CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
CUDA_ERROR_INVALID_VALUE, CUDA_ERROR_INVALID_HANDLE,
CUDA_ERROR_NOT_FOUND

###### Description

Returns in *fptr the function pointer to a unified function denoted by symbol. If no unified function
with name symbol exists, the call returns CUDA_ERROR_NOT_FOUND. If there is no device with
attribute CU_DEVICE_ATTRIBUTE_UNIFIED_FUNCTION_POINTERS present in the system, the
call may return CUDA_ERROR_NOT_FOUND.


See also:

cuLibraryLoadData, cuLibraryLoadFromFile, cuLibraryUnload
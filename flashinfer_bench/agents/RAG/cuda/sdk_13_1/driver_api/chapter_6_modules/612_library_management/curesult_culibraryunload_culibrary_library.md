# CUresult cuLibraryUnload (CUlibrary library)

Unloads a library.

###### Parameters

**library**

  - Library to unload

###### Returns

CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
CUDA_ERROR_INVALID_VALUE

###### Description

Unloads the library specified with library


CUDA Driver API TRM-06703-001 _vRelease Version  |  174


Modules


See also:

cuLibraryLoadData, cuLibraryLoadFromFile, cuModuleUnload
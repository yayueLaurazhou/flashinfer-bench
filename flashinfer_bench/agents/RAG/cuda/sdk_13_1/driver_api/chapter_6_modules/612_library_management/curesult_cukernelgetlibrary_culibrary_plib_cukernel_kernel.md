# CUresult cuKernelGetLibrary (CUlibrary *pLib, CUkernel kernel)

Returns a library handle.

###### Parameters

**pLib**

  - Returned library handle
**kernel**

  - Kernel to retrieve library handle

###### Returns

CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
CUDA_ERROR_INVALID_VALUE, CUDA_ERROR_INVALID_HANDLE,
CUDA_ERROR_NOT_FOUND


CUDA Driver API TRM-06703-001 _vRelease Version  |  161


Modules

###### Description

Returns in pLib the handle of the library for the requested kernel kernel


See also:

cuLibraryLoadData, cuLibraryLoadFromFile, cuLibraryUnload, cuLibraryGetKernel
# CUresult cuLibraryGetKernelCount (unsigned int *count, CUlibrary lib)

Returns the number of kernels within a library.

###### Parameters

**count**

  - Number of kernels found within the library
**lib**

  - Library to query

###### Returns

CUDA_SUCCESS, CUDA_ERROR_INVALID_HANDLE, CUDA_ERROR_INVALID_VALUE


CUDA Driver API TRM-06703-001 _vRelease Version  |  168


Modules

###### Description

Returns in count the number of kernels in lib.
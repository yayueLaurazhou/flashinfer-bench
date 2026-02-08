# CUresult cuLibraryEnumerateKernels (CUkernel *kernels, unsigned int numKernels, CUlibrary lib)

Retrieve the kernel handles within a library.

###### Parameters

**kernels**

  - Buffer where the kernel handles are returned to
**numKernels**

  - Maximum number of kernel handles may be returned to the buffer
**lib**

  - Library to query from

###### Returns

CUDA_SUCCESS, CUDA_ERROR_INVALID_HANDLE, CUDA_ERROR_INVALID_VALUE

###### Description

Returns in kernels a maximum number of numKernels kernel handles within lib. The returned
kernel handle becomes invalid when the library is unloaded.


CUDA Driver API TRM-06703-001 _vRelease Version  |  166


Modules


See also:

cuLibraryGetKernelCount
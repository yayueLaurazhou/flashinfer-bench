# CUresult cuLibraryGetKernel (CUkernel *pKernel, CUlibrary library, const char *name)

Returns a kernel handle.

###### Parameters

**pKernel**

  - Returned kernel handle
**library**

  - Library to retrieve kernel from
**name**

  - Name of kernel to retrieve

###### Returns

CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
CUDA_ERROR_INVALID_VALUE, CUDA_ERROR_INVALID_HANDLE,
CUDA_ERROR_NOT_FOUND

###### Description

Returns in pKernel the handle of the kernel with name name located in library library. If kernel
handle is not found, the call returns CUDA_ERROR_NOT_FOUND.


See also:

cuLibraryLoadData, cuLibraryLoadFromFile, cuLibraryUnload, cuKernelGetFunction,
cuLibraryGetModule, cuModuleGetFunction
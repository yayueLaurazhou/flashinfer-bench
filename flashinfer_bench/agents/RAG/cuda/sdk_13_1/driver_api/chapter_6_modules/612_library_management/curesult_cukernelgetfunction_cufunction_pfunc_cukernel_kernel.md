# CUresult cuKernelGetFunction (CUfunction *pFunc, CUkernel kernel)

Returns a function handle.

###### Parameters

**pFunc**

  - Returned function handle
**kernel**

  - Kernel to retrieve function for the requested context

###### Returns

CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
CUDA_ERROR_INVALID_VALUE, CUDA_ERROR_INVALID_HANDLE,
CUDA_ERROR_NOT_FOUND, CUDA_ERROR_INVALID_CONTEXT,
CUDA_ERROR_CONTEXT_IS_DESTROYED

###### Description

Returns in pFunc the handle of the function for the requested kernel kernel and the current context.
If function handle is not found, the call returns CUDA_ERROR_NOT_FOUND.


See also:

cuLibraryLoadData, cuLibraryLoadFromFile, cuLibraryUnload, cuLibraryGetKernel,
cuLibraryGetModule, cuModuleGetFunction
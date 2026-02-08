# __host__cudaError_t cudaLibraryGetKernel (cudaKernel_t *pKernel, cudaLibrary_t library, const char *name)

Returns a kernel handle.

##### Parameters

**pKernel**

  - Returned kernel handle
**library**

  - Library to retrieve kernel from
**name**

  - Name of kernel to retrieve

##### Returns

cudaSuccess, cudaErrorCudartUnloading, cudaErrorInitializationError, cudaErrorInvalidValue,
cudaErrorInvalidResourceHandle, cudaErrorSymbolNotFound

##### Description

Returns in pKernel the handle of the kernel with name name located in library library. If kernel
handle is not found, the call returns cudaErrorSymbolNotFound.


See also:


CUDA Runtime API vRelease Version  |  435


Modules


cudaLibraryLoadData, cudaLibraryLoadFromFile, cudaLibraryUnload, cuLibraryGetKernel
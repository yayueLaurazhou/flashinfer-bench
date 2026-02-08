# __host__cudaError_t cudaLibraryEnumerateKernels (cudaKernel_t *kernels, unsigned int numKernels, cudaLibrary_t lib)

Retrieve the kernel handles within a library.

##### Parameters

**kernels**

  - Buffer where the kernel handles are returned to
**numKernels**

  - Maximum number of kernel handles may be returned to the buffer
**lib**

  - Library to query from

##### Returns

cudaSuccess, cudaErrorCudartUnloading, cudaErrorInitializationError, cudaErrorInvalidValue,
cudaErrorInvalidResourceHandle

##### Description

Returns in kernels a maximum number of numKernels kernel handles within lib. The returned
kernel handle becomes invalid when the library is unloaded.


See also:

cudaLibraryGetKernelCount, cuLibraryEnumerateKernels
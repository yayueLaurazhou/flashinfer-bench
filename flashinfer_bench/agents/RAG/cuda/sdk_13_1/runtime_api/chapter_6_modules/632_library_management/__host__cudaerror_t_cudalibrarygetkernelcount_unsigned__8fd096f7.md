# __host__cudaError_t cudaLibraryGetKernelCount (unsigned int *count, cudaLibrary_t lib)

Returns the number of kernels within a library.

##### Parameters

**count**

  - Number of kernels found within the library
**lib**

  - Library to query

##### Returns

cudaSuccess, cudaErrorCudartUnloading, cudaErrorInitializationError, cudaErrorInvalidValue,
cudaErrorInvalidResourceHandle

##### Description

Returns in count the number of kernels in lib.


See also:

cudaLibraryEnumerateKernels, cudaLibraryLoadFromFile, cudaLibraryLoadData,
cuLibraryGetKernelCount
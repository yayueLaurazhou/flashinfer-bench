# __host__cudaError_t cudaLibraryUnload (cudaLibrary_t library)

Unloads a library.

##### Parameters

**library**

  - Library to unload

##### Returns

cudaSuccess, cudaErrorCudartUnloading, cudaErrorInitializationError, cudaErrorInvalidValue

##### Description

Unloads the library specified with library


See also:

cudaLibraryLoadData, cudaLibraryLoadFromFile, cuLibraryUnload
# template < class T > __host__cudaError_t cudaLibraryGetGlobal (T **dptr, size_t *bytes, cudaLibrary_t library, const char *name)

Returns a global device pointer.

##### Parameters

**dptr**

  - Returned global device pointer for the requested library
**bytes**

  - Returned global size in bytes
**library**

  - Library to retrieve global from
**name**

  - Name of global to retrieve

##### Returns

cudaSuccess, cudaErrorCudartUnloading, cudaErrorInitializationError, cudaErrorInvalidValue,
cudaErrorInvalidResourceHandle, cudaErrorSymbolNotFound cudaErrorDeviceUninitialized,
cudaErrorContextIsDestroyed

##### Description

Returns in *dptr and *bytes the base pointer and size of the global with name name for the
requested library library and the current device. If no global for the requested name name exists,
the call returns cudaErrorSymbolNotFound. One of the parameters dptr or bytes (not both) can be
NULL in which case it is ignored.


See also:

cudaLibraryLoadData, cudaLibraryLoadFromFile, cudaLibraryUnload, cudaLibraryGetManaged


CUDA Runtime API vRelease Version  |  487


Modules
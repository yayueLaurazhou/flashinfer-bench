# __host__cudaError_t cudaLibraryGetGlobal (void **dptr, size_t *bytes, cudaLibrary_t library, const char *name)

Returns a global device pointer.

##### Parameters

**dptr**

  - Returned global device pointer for the requested library
**bytes**

  - Returned global size in bytes
**library**

  - Library to retrieve global from


CUDA Runtime API vRelease Version  |  434


Modules


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
NULL in which case it is ignored. The returned dptr cannot be passed to the Symbol APIs such as
cudaMemcpyToSymbol, cudaMemcpyFromSymbol, cudaGetSymbolAddress, or cudaGetSymbolSize.


See also:

cudaLibraryLoadData, cudaLibraryLoadFromFile, cudaLibraryUnload, cudaLibraryGetManaged,
cuLibraryGetGlobal
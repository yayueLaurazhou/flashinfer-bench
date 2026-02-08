# __host__cudaError_t cudaLibraryGetManaged (void **dptr, size_t *bytes, cudaLibrary_t library, const char *name)

Returns a pointer to managed memory.

##### Parameters

**dptr**

  - Returned pointer to the managed memory
**bytes**

  - Returned memory size in bytes
**library**

  - Library to retrieve managed memory from
**name**

  - Name of managed memory to retrieve


CUDA Runtime API vRelease Version  |  436


Modules

##### Returns

cudaSuccess, cudaErrorCudartUnloading, cudaErrorInitializationError, cudaErrorInvalidValue,
cudaErrorInvalidResourceHandle, cudaErrorSymbolNotFound

##### Description

Returns in *dptr and *bytes the base pointer and size of the managed memory with name name
for the requested library library. If no managed memory with the requested name name exists,
the call returns cudaErrorSymbolNotFound. One of the parameters dptr or bytes (not both) can
be NULL in which case it is ignored. Note that managed memory for library library is shared
across devices and is registered when the library is loaded. The returned dptr cannot be passed to the
Symbol APIs such as cudaMemcpyToSymbol, cudaMemcpyFromSymbol, cudaGetSymbolAddress, or
cudaGetSymbolSize.


See also:

cudaLibraryLoadData, cudaLibraryLoadFromFile, cudaLibraryUnload, cudaLibraryGetGlobal,
cuLibraryGetManaged
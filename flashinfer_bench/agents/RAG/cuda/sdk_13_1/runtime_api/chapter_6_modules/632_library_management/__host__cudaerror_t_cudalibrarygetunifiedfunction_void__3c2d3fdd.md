# __host__cudaError_t cudaLibraryGetUnifiedFunction (void **fptr, cudaLibrary_t library, const char *symbol)

Returns a pointer to a unified function.

##### Parameters

**fptr**

  - Returned pointer to a unified function
**library**

  - Library to retrieve function pointer memory from
**symbol**

  - Name of function pointer to retrieve

##### Returns

cudaSuccess, cudaErrorCudartUnloading, cudaErrorInitializationError, cudaErrorInvalidValue,
cudaErrorInvalidResourceHandle, cudaErrorSymbolNotFound

##### Description

Returns in *fptr the function pointer to a unified function denoted by symbol. If no unified
function with name symbol exists, the call returns cudaErrorSymbolNotFound. If there is no device
with attribute cudaDeviceProp::unifiedFunctionPointers present in the system, the call may return
cudaErrorSymbolNotFound.


CUDA Runtime API vRelease Version  |  437


Modules


See also:

cudaLibraryLoadData, cudaLibraryLoadFromFile, cudaLibraryUnload, cuLibraryGetUnifiedFunction
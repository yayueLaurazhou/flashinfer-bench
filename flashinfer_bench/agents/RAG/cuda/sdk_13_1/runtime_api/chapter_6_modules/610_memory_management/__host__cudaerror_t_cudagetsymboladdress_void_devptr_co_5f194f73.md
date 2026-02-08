# __host__cudaError_t cudaGetSymbolAddress (void **devPtr, const void *symbol)

Finds the address associated with a CUDA symbol.

##### Parameters

**devPtr**

  - Return device pointer associated with symbol
**symbol**

  - Device symbol address

##### Returns

cudaSuccess, cudaErrorInvalidSymbol, cudaErrorNoKernelImageForDevice

##### Description

Returns in *devPtr the address of symbol symbol on the device. symbol is a variable that resides
in global or constant memory space. If symbol cannot be found, or if symbol is not declared in the
global or constant memory space, *devPtr is unchanged and the error cudaErrorInvalidSymbol is
returned.









See also:

cudaGetSymbolAddress ( C++ API), cudaGetSymbolSize ( C API), cuModuleGetGlobal


CUDA Runtime API vRelease Version  |  126


Modules
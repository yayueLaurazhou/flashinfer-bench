# __host__cudaError_t cudaGetSymbolSize (size_t *size, const void *symbol)

Finds the size of the object associated with a CUDA symbol.

##### Parameters

**size**

  - Size of object associated with symbol
**symbol**

  - Device symbol address

##### Returns

cudaSuccess, cudaErrorInvalidSymbol, cudaErrorNoKernelImageForDevice

##### Description

Returns in *size the size of symbol symbol. symbol is a variable that resides in global or constant
memory space. If symbol cannot be found, or if symbol is not declared in global or constant
memory space, *size is unchanged and the error cudaErrorInvalidSymbol is returned.









See also:

cudaGetSymbolAddress ( C API), cudaGetSymbolSize ( C++ API), cuModuleGetGlobal


CUDA Runtime API vRelease Version  |  127


Modules
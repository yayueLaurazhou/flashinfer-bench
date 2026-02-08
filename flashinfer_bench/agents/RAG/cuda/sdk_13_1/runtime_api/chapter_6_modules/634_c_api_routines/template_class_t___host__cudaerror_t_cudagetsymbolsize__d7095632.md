# template < class T > __host__cudaError_t cudaGetSymbolSize (size_t *size, const T symbol)

[C++ API] Finds the size of the object associated with a CUDA symbol

##### Parameters

**size**

  - Size of object associated with symbol
**symbol**

  - Device symbol reference

##### Returns

cudaSuccess, cudaErrorInvalidSymbol, cudaErrorNoKernelImageForDevice

##### Description

Returns in *size the size of symbol symbol. symbol must be a variable that resides in global
or constant memory space. If symbol cannot be found, or if symbol is not declared in global or
constant memory space, *size is unchanged and the error cudaErrorInvalidSymbol is returned.


Note:

**‣** Note that this function may also return error codes from previous, asynchronous launches.


CUDA Runtime API vRelease Version  |  470


Modules





See also:

cudaGetSymbolAddress ( C++ API), cudaGetSymbolSize ( C API)
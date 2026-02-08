# template < class T > __host__cudaError_t cudaGetSymbolAddress (void **devPtr, const T symbol)

[C++ API] Finds the address associated with a CUDA symbol

##### Parameters

**devPtr**

  - Return device pointer associated with symbol
**symbol**

  - Device symbol reference

##### Returns

cudaSuccess, cudaErrorInvalidSymbol, cudaErrorNoKernelImageForDevice


CUDA Runtime API vRelease Version  |  469


Modules

##### Description

Returns in *devPtr the address of symbol symbol on the device. symbol can either be a
variable that resides in global or constant memory space. If symbol cannot be found, or if symbol
is not declared in the global or constant memory space, *devPtr is unchanged and the error
cudaErrorInvalidSymbol is returned.



See also:

cudaGetSymbolAddress ( C API), cudaGetSymbolSize ( C++ API)
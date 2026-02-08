# template < class T > __host__cudaError_t cudaMemcpyFromSymbol (void *dst, const T symbol, size_t count, size_t offset, cudaMemcpyKind kind)

[C++ API] Copies data from the given symbol on the device

##### Parameters

**dst**

  - Destination memory address
**symbol**

  - Device symbol reference
**count**

  - Size in bytes to copy
**offset**

  - Offset from start of symbol in bytes
**kind**

  - Type of transfer

##### Returns

cudaSuccess, cudaErrorInvalidValue, cudaErrorInvalidSymbol, cudaErrorInvalidMemcpyDirection,
cudaErrorNoKernelImageForDevice

##### Description

Copies count bytes from the memory area offset bytes from the start of symbol symbol to
the memory area pointed to by dst. The memory areas may not overlap. symbol is a variable


CUDA Runtime API vRelease Version  |  494


Modules


that resides in global or constant memory space. kind can be either cudaMemcpyDeviceToHost or
cudaMemcpyDeviceToDevice.













See also:

cudaMemcpy, cudaMemcpy2D, cudaMemcpy2DToArray, cudaMemcpy2DFromArray,
cudaMemcpy2DArrayToArray, cudaMemcpyToSymbol, cudaMemcpyAsync, cudaMemcpy2DAsync,
cudaMemcpy2DToArrayAsync, cudaMemcpy2DFromArrayAsync, cudaMemcpyToSymbolAsync,
cudaMemcpyFromSymbolAsync
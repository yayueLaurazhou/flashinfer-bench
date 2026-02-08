# template < class T > __host__cudaError_t cudaMemcpyToSymbol (const T symbol, const void *src, size_t count, size_t offset, cudaMemcpyKind kind)

[C++ API] Copies data to the given symbol on the device

##### Parameters

**symbol**

  - Device symbol reference
**src**

  - Source memory address
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

Copies count bytes from the memory area pointed to by src to the memory area offset bytes
from the start of symbol symbol. The memory areas may not overlap. symbol is a variable that
resides in global or constant memory space. kind can be either cudaMemcpyHostToDevice or
cudaMemcpyDeviceToDevice.













See also:


CUDA Runtime API vRelease Version  |  497


Modules


cudaMemcpy, cudaMemcpy2D, cudaMemcpy2DToArray, cudaMemcpy2DFromArray,
cudaMemcpy2DArrayToArray, cudaMemcpyFromSymbol, cudaMemcpyAsync,
cudaMemcpy2DAsync, cudaMemcpy2DToArrayAsync, cudaMemcpy2DFromArrayAsync,
cudaMemcpyToSymbolAsync, cudaMemcpyFromSymbolAsync
# __host__cudaError_t cudaMemcpyToSymbol (const void *symbol, const void *src, size_t count, size_t offset, cudaMemcpyKind kind)

Copies data to the given symbol on the device.

##### Parameters

**symbol**

  - Device symbol address
**src**

  - Source memory address
**count**

  - Size in bytes to copy
**offset**

  - Offset from start of symbol in bytes
**kind**

  - Type of transfer


CUDA Runtime API vRelease Version  |  179


Modules

##### Returns

cudaSuccess, cudaErrorInvalidValue, cudaErrorInvalidSymbol, cudaErrorInvalidMemcpyDirection,
cudaErrorNoKernelImageForDevice

##### Description

Copies count bytes from the memory area pointed to by src to the memory area pointed
to by offset bytes from the start of symbol symbol. The memory areas may not overlap.
symbol is a variable that resides in global or constant memory space. kind can be either
cudaMemcpyHostToDevice, cudaMemcpyDeviceToDevice, or cudaMemcpyDefault. Passing
cudaMemcpyDefault is recommended, in which case the type of transfer is inferred from the pointer
values. However, cudaMemcpyDefault is only allowed on systems that support unified virtual
addressing.













See also:

cudaMemcpy, cudaMemcpy2D, cudaMemcpy2DToArray, cudaMemcpy2DFromArray,
cudaMemcpy2DArrayToArray, cudaMemcpyFromSymbol, cudaMemcpyAsync,
cudaMemcpy2DAsync, cudaMemcpy2DToArrayAsync, cudaMemcpy2DFromArrayAsync,
cudaMemcpyToSymbolAsync, cudaMemcpyFromSymbolAsync, cuMemcpy, cuMemcpyHtoD,
cuMemcpyDtoD


CUDA Runtime API vRelease Version  |  180


Modules
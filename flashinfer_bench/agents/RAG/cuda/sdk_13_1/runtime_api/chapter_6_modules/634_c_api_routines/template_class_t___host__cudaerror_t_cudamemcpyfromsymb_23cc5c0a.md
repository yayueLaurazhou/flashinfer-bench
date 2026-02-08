# template < class T > __host__cudaError_t cudaMemcpyFromSymbolAsync (void *dst, const T symbol, size_t count, size_t offset, cudaMemcpyKind kind, cudaStream_t stream)

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
**stream**

  - Stream identifier


CUDA Runtime API vRelease Version  |  495


Modules

##### Returns

cudaSuccess, cudaErrorInvalidValue, cudaErrorInvalidSymbol, cudaErrorInvalidMemcpyDirection,
cudaErrorNoKernelImageForDevice

##### Description

Copies count bytes from the memory area offset bytes from the start of symbol symbol to
the memory area pointed to by dst. The memory areas may not overlap. symbol is a variable
that resides in global or constant memory space. kind can be either cudaMemcpyDeviceToHost or
cudaMemcpyDeviceToDevice.

cudaMemcpyFromSymbolAsync() is asynchronous with respect to the host, so the call may return
before the copy is complete. The copy can optionally be associated to a stream by passing a non-zero
stream argument. If kind is cudaMemcpyDeviceToHost and stream is non-zero, the copy may
overlap with operations in other streams.











See also:

cudaMemcpy, cudaMemcpy2D, cudaMemcpy2DToArray, cudaMemcpy2DFromArray,
cudaMemcpy2DArrayToArray, cudaMemcpyToSymbol, cudaMemcpyFromSymbol,
cudaMemcpyAsync, cudaMemcpy2DAsync, cudaMemcpy2DToArrayAsync,
cudaMemcpy2DFromArrayAsync, cudaMemcpyToSymbolAsync


CUDA Runtime API vRelease Version  |  496


Modules
# __host__cudaError_t cudaMemcpyToSymbolAsync (const void *symbol, const void *src, size_t count, size_t offset, cudaMemcpyKind kind, cudaStream_t stream)

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
**stream**

  - Stream identifier

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

cudaMemcpyToSymbolAsync() is asynchronous with respect to the host, so the call may return before
the copy is complete. The copy can optionally be associated to a stream by passing a non-zero stream
argument. If kind is cudaMemcpyHostToDevice and stream is non-zero, the copy may overlap with
operations in other streams.


Note:

**‣** Note that this function may also return error codes from previous, asynchronous launches.


CUDA Runtime API vRelease Version  |  181


Modules











See also:

cudaMemcpy, cudaMemcpy2D, cudaMemcpy2DToArray, cudaMemcpy2DFromArray,
cudaMemcpy2DArrayToArray, cudaMemcpyToSymbol, cudaMemcpyFromSymbol,
cudaMemcpyAsync, cudaMemcpy2DAsync, cudaMemcpy2DToArrayAsync,
cudaMemcpy2DFromArrayAsync, cudaMemcpyFromSymbolAsync, cuMemcpyAsync,
cuMemcpyHtoDAsync, cuMemcpyDtoDAsync
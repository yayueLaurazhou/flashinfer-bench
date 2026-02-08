# __host____device__cudaError_t cudaMemcpyAsync (void *dst, const void *src, size_t count, cudaMemcpyKind kind, cudaStream_t stream)

Copies data between host and device.

##### Parameters

**dst**

  - Destination memory address
**src**

  - Source memory address
**count**

  - Size in bytes to copy
**kind**

  - Type of transfer
**stream**

  - Stream identifier

##### Returns

cudaSuccess, cudaErrorInvalidValue, cudaErrorInvalidMemcpyDirection

##### Description

Copies count bytes from the memory area pointed to by src to the memory area pointed to by
dst, where kind specifies the direction of the copy, and must be one of cudaMemcpyHostToHost,
cudaMemcpyHostToDevice, cudaMemcpyDeviceToHost, cudaMemcpyDeviceToDevice, or
cudaMemcpyDefault. Passing cudaMemcpyDefault is recommended, in which case the type of transfer
is inferred from the pointer values. However, cudaMemcpyDefault is only allowed on systems that
support unified virtual addressing.

The memory areas may not overlap. Calling cudaMemcpyAsync() with dst and src pointers that do
not match the direction of the copy results in an undefined behavior.


CUDA Runtime API vRelease Version  |  171


Modules


cudaMemcpyAsync() is asynchronous with respect to the host, so the call may return before the copy is
complete. The copy can optionally be associated to a stream by passing a non-zero stream argument.
If kind is cudaMemcpyHostToDevice or cudaMemcpyDeviceToHost and the stream is non-zero,
the copy may overlap with operations in other streams.

The device version of this function only handles device to device copies and cannot be given local or
shared pointers.













See also:

cudaMemcpy, cudaMemcpy2D, cudaMemcpy2DToArray, cudaMemcpy2DFromArray,
cudaMemcpy2DArrayToArray, cudaMemcpyToSymbol, cudaMemcpyFromSymbol,
cudaMemcpy2DAsync, cudaMemcpy2DToArrayAsync, cudaMemcpy2DFromArrayAsync,
cudaMemcpyToSymbolAsync, cudaMemcpyFromSymbolAsync, cuMemcpyAsync,
cuMemcpyDtoHAsync, cuMemcpyHtoDAsync, cuMemcpyDtoDAsync
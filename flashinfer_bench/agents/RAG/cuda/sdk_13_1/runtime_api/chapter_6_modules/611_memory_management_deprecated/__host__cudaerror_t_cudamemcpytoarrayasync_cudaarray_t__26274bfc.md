# __host__cudaError_t cudaMemcpyToArrayAsync (cudaArray_t dst, size_t wOffset, size_t hOffset, const void *src, size_t count, cudaMemcpyKind kind, cudaStream_t stream)

Copies data between host and device.

##### Parameters

**dst**

  - Destination memory address
**wOffset**

  - Destination starting X offset (columns in bytes)
**hOffset**

  - Destination starting Y offset (rows)
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

Deprecated

Copies count bytes from the memory area pointed to by src to the CUDA array dst starting
at hOffset rows and wOffset bytes from the upper left corner, where kind specifies the
direction of the copy, and must be one of cudaMemcpyHostToHost, cudaMemcpyHostToDevice,
cudaMemcpyDeviceToHost, cudaMemcpyDeviceToDevice, or cudaMemcpyDefault. Passing
cudaMemcpyDefault is recommended, in which case the type of transfer is inferred from the pointer
values. However, cudaMemcpyDefault is only allowed on systems that support unified virtual
addressing.

cudaMemcpyToArrayAsync() is asynchronous with respect to the host, so the call may return before
the copy is complete. The copy can optionally be associated to a stream by passing a non-zero stream
argument. If kind is cudaMemcpyHostToDevice or cudaMemcpyDeviceToHost and stream is nonzero, the copy may overlap with operations in other streams.


CUDA Runtime API vRelease Version  |  209


Modules













See also:

cudaMemcpy, cudaMemcpy2D, cudaMemcpyToArray, cudaMemcpy2DToArray,
cudaMemcpyFromArray, cudaMemcpy2DFromArray, cudaMemcpyArrayToArray,
cudaMemcpy2DArrayToArray, cudaMemcpyToSymbol, cudaMemcpyFromSymbol,
cudaMemcpyAsync, cudaMemcpy2DAsync, cudaMemcpy2DToArrayAsync,
cudaMemcpyFromArrayAsync, cudaMemcpy2DFromArrayAsync, cudaMemcpyToSymbolAsync,
cudaMemcpyFromSymbolAsync, cuMemcpyHtoAAsync, cuMemcpy2DAsync
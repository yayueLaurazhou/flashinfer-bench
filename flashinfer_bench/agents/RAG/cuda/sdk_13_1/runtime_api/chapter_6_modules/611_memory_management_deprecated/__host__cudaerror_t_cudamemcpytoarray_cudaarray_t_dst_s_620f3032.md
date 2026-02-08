# __host__cudaError_t cudaMemcpyToArray (cudaArray_t dst, size_t wOffset, size_t hOffset, const void *src, size_t count, cudaMemcpyKind kind)

Copies data between host and device.

##### Parameters

**dst**

  - Destination memory address


CUDA Runtime API vRelease Version  |  207


Modules


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











See also:

cudaMemcpy, cudaMemcpy2D, cudaMemcpy2DToArray, cudaMemcpyFromArray,
cudaMemcpy2DFromArray, cudaMemcpyArrayToArray, cudaMemcpy2DArrayToArray,
cudaMemcpyToSymbol, cudaMemcpyFromSymbol, cudaMemcpyAsync, cudaMemcpy2DAsync,
cudaMemcpyToArrayAsync, cudaMemcpy2DToArrayAsync, cudaMemcpyFromArrayAsync,
cudaMemcpy2DFromArrayAsync, cudaMemcpyToSymbolAsync, cudaMemcpyFromSymbolAsync,
cuMemcpyHtoA, cuMemcpyDtoA


CUDA Runtime API vRelease Version  |  208


Modules
# __host__cudaError_t cudaMemcpyArrayToArray (cudaArray_t dst, size_t wOffsetDst, size_t hOffsetDst, cudaArray_const_t src, size_t wOffsetSrc, size_t hOffsetSrc, size_t count, cudaMemcpyKind kind)

Copies data between host and device.

##### Parameters

**dst**

  - Destination memory address


CUDA Runtime API vRelease Version  |  203


Modules


**wOffsetDst**

  - Destination starting X offset (columns in bytes)
**hOffsetDst**

  - Destination starting Y offset (rows)
**src**

  - Source memory address
**wOffsetSrc**

  - Source starting X offset (columns in bytes)
**hOffsetSrc**

  - Source starting Y offset (rows)
**count**

  - Size in bytes to copy
**kind**

  - Type of transfer

##### Returns

cudaSuccess, cudaErrorInvalidValue, cudaErrorInvalidMemcpyDirection

##### Description

Deprecated

Copies count bytes from the CUDA array src starting at hOffsetSrc rows and wOffsetSrc
bytes from the upper left corner to the CUDA array dst starting at hOffsetDst rows and
wOffsetDst bytes from the upper left corner, where kind specifies the direction of the copy, and
must be one of cudaMemcpyHostToHost, cudaMemcpyHostToDevice, cudaMemcpyDeviceToHost,
cudaMemcpyDeviceToDevice, or cudaMemcpyDefault. Passing cudaMemcpyDefault is
recommended, in which case the type of transfer is inferred from the pointer values. However,
cudaMemcpyDefault is only allowed on systems that support unified virtual addressing.



See also:

cudaMemcpy, cudaMemcpy2D, cudaMemcpyToArray, cudaMemcpy2DToArray,
cudaMemcpyFromArray, cudaMemcpy2DFromArray, cudaMemcpy2DArrayToArray,
cudaMemcpyToSymbol, cudaMemcpyFromSymbol, cudaMemcpyAsync, cudaMemcpy2DAsync,
cudaMemcpyToArrayAsync, cudaMemcpy2DToArrayAsync, cudaMemcpyFromArrayAsync,


CUDA Runtime API vRelease Version  |  204


Modules


cudaMemcpy2DFromArrayAsync, cudaMemcpyToSymbolAsync, cudaMemcpyFromSymbolAsync,
cuMemcpyAtoA
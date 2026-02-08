# __host__cudaError_t cudaMemcpy2DArrayToArray (cudaArray_t dst, size_t wOffsetDst, size_t hOffsetDst, cudaArray_const_t src, size_t wOffsetSrc, size_t hOffsetSrc, size_t width, size_t height, cudaMemcpyKind kind)

Copies data between host and device.

##### Parameters

**dst**

  - Destination memory address
**wOffsetDst**

  - Destination starting X offset (columns in bytes)
**hOffsetDst**

  - Destination starting Y offset (rows)
**src**

  - Source memory address


CUDA Runtime API vRelease Version  |  153


Modules


**wOffsetSrc**

  - Source starting X offset (columns in bytes)
**hOffsetSrc**

  - Source starting Y offset (rows)
**width**

  - Width of matrix transfer (columns in bytes)
**height**

  - Height of matrix transfer (rows)
**kind**

  - Type of transfer

##### Returns

cudaSuccess, cudaErrorInvalidValue, cudaErrorInvalidMemcpyDirection

##### Description

Copies a matrix (height rows of width bytes each) from the CUDA array src starting at
hOffsetSrc rows and wOffsetSrc bytes from the upper left corner to the CUDA array
dst starting at hOffsetDst rows and wOffsetDst bytes from the upper left corner,
where kind specifies the direction of the copy, and must be one of cudaMemcpyHostToHost,
cudaMemcpyHostToDevice, cudaMemcpyDeviceToHost, cudaMemcpyDeviceToDevice, or
cudaMemcpyDefault. Passing cudaMemcpyDefault is recommended, in which case the type of transfer
is inferred from the pointer values. However, cudaMemcpyDefault is only allowed on systems that
support unified virtual addressing. wOffsetDst + width must not exceed the width of the CUDA
array dst. wOffsetSrc + width must not exceed the width of the CUDA array src.











See also:

cudaMemcpy, cudaMemcpy2D, cudaMemcpy2DToArray, cudaMemcpy2DFromArray,
cudaMemcpyToSymbol, cudaMemcpyFromSymbol, cudaMemcpyAsync, cudaMemcpy2DAsync,
cudaMemcpy2DToArrayAsync, cudaMemcpy2DFromArrayAsync, cudaMemcpyToSymbolAsync,
cudaMemcpyFromSymbolAsync, cuMemcpy2D, cuMemcpy2DUnaligned


CUDA Runtime API vRelease Version  |  154


Modules
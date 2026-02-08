# __host__cudaError_t cudaMemcpy2DToArray (cudaArray_t dst, size_t wOffset, size_t hOffset, const void *src, size_t spitch, size_t width, size_t height, cudaMemcpyKind kind)

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
**spitch**

  - Pitch of source memory
**width**

  - Width of matrix transfer (columns in bytes)
**height**

  - Height of matrix transfer (rows)
**kind**

  - Type of transfer

##### Returns

cudaSuccess, cudaErrorInvalidValue, cudaErrorInvalidPitchValue, cudaErrorInvalidMemcpyDirection

##### Description

Copies a matrix (height rows of width bytes each) from the memory area pointed to by src to
the CUDA array dst starting at hOffset rows and wOffset bytes from the upper left corner,
where kind specifies the direction of the copy, and must be one of cudaMemcpyHostToHost,
cudaMemcpyHostToDevice, cudaMemcpyDeviceToHost, cudaMemcpyDeviceToDevice, or
cudaMemcpyDefault. Passing cudaMemcpyDefault is recommended, in which case the type of transfer
is inferred from the pointer values. However, cudaMemcpyDefault is only allowed on systems that
support unified virtual addressing. spitch is the width in memory in bytes of the 2D array pointed to


CUDA Runtime API vRelease Version  |  160


Modules


by src, including any padding added to the end of each row. wOffset + width must not exceed the
width of the CUDA array dst. width must not exceed spitch. cudaMemcpy2DToArray() returns
an error if spitch exceeds the maximum allowed.











See also:

cudaMemcpy, cudaMemcpy2D, cudaMemcpy2DFromArray, cudaMemcpy2DArrayToArray,
cudaMemcpyToSymbol, cudaMemcpyFromSymbol, cudaMemcpyAsync, cudaMemcpy2DAsync,
cudaMemcpy2DToArrayAsync, cudaMemcpy2DFromArrayAsync, cudaMemcpyToSymbolAsync,
cudaMemcpyFromSymbolAsync, cuMemcpy2D, cuMemcpy2DUnaligned
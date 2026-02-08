# __host__cudaError_t cudaMemcpy2DFromArray (void *dst, size_t dpitch, cudaArray_const_t src, size_t wOffset, size_t hOffset, size_t width, size_t height, cudaMemcpyKind kind)

Copies data between host and device.

##### Parameters

**dst**

  - Destination memory address
**dpitch**

  - Pitch of destination memory
**src**

  - Source memory address
**wOffset**

  - Source starting X offset (columns in bytes)
**hOffset**

  - Source starting Y offset (rows)
**width**

  - Width of matrix transfer (columns in bytes)
**height**

  - Height of matrix transfer (rows)
**kind**

  - Type of transfer

##### Returns

cudaSuccess, cudaErrorInvalidValue, cudaErrorInvalidPitchValue, cudaErrorInvalidMemcpyDirection

##### Description

Copies a matrix (height rows of width bytes each) from the CUDA array src starting at
hOffset rows and wOffset bytes from the upper left corner to the memory area pointed to by
dst, where kind specifies the direction of the copy, and must be one of cudaMemcpyHostToHost,
cudaMemcpyHostToDevice, cudaMemcpyDeviceToHost, cudaMemcpyDeviceToDevice, or
cudaMemcpyDefault. Passing cudaMemcpyDefault is recommended, in which case the type of transfer
is inferred from the pointer values. However, cudaMemcpyDefault is only allowed on systems that
support unified virtual addressing. dpitch is the width in memory in bytes of the 2D array pointed
to by dst, including any padding added to the end of each row. wOffset + width must not exceed
the width of the CUDA array src. width must not exceed dpitch. cudaMemcpy2DFromArray()
returns an error if dpitch exceeds the maximum allowed.


CUDA Runtime API vRelease Version  |  157


Modules











See also:

cudaMemcpy, cudaMemcpy2D, cudaMemcpy2DToArray, cudaMemcpy2DArrayToArray,
cudaMemcpyToSymbol, cudaMemcpyFromSymbol, cudaMemcpyAsync, cudaMemcpy2DAsync,
cudaMemcpy2DToArrayAsync, cudaMemcpy2DFromArrayAsync, cudaMemcpyToSymbolAsync,
cudaMemcpyFromSymbolAsync, cuMemcpy2D, cuMemcpy2DUnaligned
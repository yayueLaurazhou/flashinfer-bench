# __host__cudaError_t cudaMemcpy2DFromArrayAsync (void *dst, size_t dpitch, cudaArray_const_t src, size_t wOffset, size_t hOffset, size_t width, size_t height, cudaMemcpyKind kind, cudaStream_t stream)

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


CUDA Runtime API vRelease Version  |  158


Modules


**height**

  - Height of matrix transfer (rows)
**kind**

  - Type of transfer
**stream**

  - Stream identifier

##### Returns

cudaSuccess, cudaErrorInvalidValue, cudaErrorInvalidPitchValue, cudaErrorInvalidMemcpyDirection

##### Description

Copies a matrix (height rows of width bytes each) from the CUDA array src starting at
hOffset rows and wOffset bytes from the upper left corner to the memory area pointed to by
dst, where kind specifies the direction of the copy, and must be one of cudaMemcpyHostToHost,
cudaMemcpyHostToDevice, cudaMemcpyDeviceToHost, cudaMemcpyDeviceToDevice, or
cudaMemcpyDefault. Passing cudaMemcpyDefault is recommended, in which case the type of transfer
is inferred from the pointer values. However, cudaMemcpyDefault is only allowed on systems that
support unified virtual addressing. dpitch is the width in memory in bytes of the 2D array pointed to
by dst, including any padding added to the end of each row. wOffset + width must not exceed the
width of the CUDA array src. width must not exceed dpitch. cudaMemcpy2DFromArrayAsync()
returns an error if dpitch exceeds the maximum allowed.

cudaMemcpy2DFromArrayAsync() is asynchronous with respect to the host, so the call may return
before the copy is complete. The copy can optionally be associated to a stream by passing a nonzero stream argument. If kind is cudaMemcpyHostToDevice or cudaMemcpyDeviceToHost and
stream is non-zero, the copy may overlap with operations in other streams.













See also:


CUDA Runtime API vRelease Version  |  159


Modules


cudaMemcpy, cudaMemcpy2D, cudaMemcpy2DToArray, cudaMemcpy2DFromArray,
cudaMemcpy2DArrayToArray, cudaMemcpyToSymbol, cudaMemcpyFromSymbol,
cudaMemcpyAsync, cudaMemcpy2DAsync, cudaMemcpy2DToArrayAsync,

cudaMemcpyToSymbolAsync, cudaMemcpyFromSymbolAsync, cuMemcpy2DAsync
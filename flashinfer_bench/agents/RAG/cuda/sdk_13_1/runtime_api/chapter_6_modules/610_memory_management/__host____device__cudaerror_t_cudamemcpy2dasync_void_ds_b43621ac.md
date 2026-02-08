# __host____device__cudaError_t cudaMemcpy2DAsync (void *dst, size_t dpitch, const void *src, size_t spitch, size_t width, size_t height, cudaMemcpyKind kind, cudaStream_t stream)

Copies data between host and device.

##### Parameters

**dst**

  - Destination memory address
**dpitch**

  - Pitch of destination memory
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
**stream**

  - Stream identifier

##### Returns

cudaSuccess, cudaErrorInvalidValue, cudaErrorInvalidPitchValue, cudaErrorInvalidMemcpyDirection

##### Description

Copies a matrix (height rows of width bytes each) from the memory area pointed to by src
to the memory area pointed to by dst, where kind specifies the direction of the copy, and must
be one of cudaMemcpyHostToHost, cudaMemcpyHostToDevice, cudaMemcpyDeviceToHost,
cudaMemcpyDeviceToDevice, or cudaMemcpyDefault. Passing cudaMemcpyDefault is
recommended, in which case the type of transfer is inferred from the pointer values. However,
cudaMemcpyDefault is only allowed on systems that support unified virtual addressing. dpitch and
spitch are the widths in memory in bytes of the 2D arrays pointed to by dst and src, including any
padding added to the end of each row. The memory areas may not overlap. width must not exceed
either dpitch or spitch.


CUDA Runtime API vRelease Version  |  155


Modules


Calling cudaMemcpy2DAsync() with dst and src pointers that do not match the direction of the
copy results in an undefined behavior. cudaMemcpy2DAsync() returns an error if dpitch or spitch
is greater than the maximum allowed.

cudaMemcpy2DAsync() is asynchronous with respect to the host, so the call may return before the
copy is complete. The copy can optionally be associated to a stream by passing a non-zero stream
argument. If kind is cudaMemcpyHostToDevice or cudaMemcpyDeviceToHost and stream is nonzero, the copy may overlap with operations in other streams.

The device version of this function only handles device to device copies and cannot be given local or
shared pointers.













See also:

cudaMemcpy, cudaMemcpy2D, cudaMemcpy2DToArray, cudaMemcpy2DFromArray,
cudaMemcpy2DArrayToArray, cudaMemcpyToSymbol, cudaMemcpyFromSymbol,
cudaMemcpyAsync, cudaMemcpy2DToArrayAsync, cudaMemcpy2DFromArrayAsync,
cudaMemcpyToSymbolAsync, cudaMemcpyFromSymbolAsync, cuMemcpy2DAsync


CUDA Runtime API vRelease Version  |  156


Modules
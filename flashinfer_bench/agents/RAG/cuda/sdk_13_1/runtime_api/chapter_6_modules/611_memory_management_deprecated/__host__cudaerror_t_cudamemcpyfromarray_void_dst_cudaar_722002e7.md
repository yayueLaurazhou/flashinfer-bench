# __host__cudaError_t cudaMemcpyFromArray (void *dst, cudaArray_const_t src, size_t wOffset, size_t hOffset, size_t count, cudaMemcpyKind kind)

Copies data between host and device.

##### Parameters

**dst**

  - Destination memory address
**src**

  - Source memory address
**wOffset**

  - Source starting X offset (columns in bytes)
**hOffset**

  - Source starting Y offset (rows)
**count**

  - Size in bytes to copy
**kind**

  - Type of transfer

##### Returns

cudaSuccess, cudaErrorInvalidValue, cudaErrorInvalidMemcpyDirection

##### Description

Deprecated

Copies count bytes from the CUDA array src starting at hOffset rows and wOffset bytes
from the upper left corner to the memory area pointed to by dst, where kind specifies the
direction of the copy, and must be one of cudaMemcpyHostToHost, cudaMemcpyHostToDevice,
cudaMemcpyDeviceToHost, cudaMemcpyDeviceToDevice, or cudaMemcpyDefault. Passing
cudaMemcpyDefault is recommended, in which case the type of transfer is inferred from the pointer
values. However, cudaMemcpyDefault is only allowed on systems that support unified virtual
addressing.









CUDA Runtime API vRelease Version  |  205


Modules





See also:

cudaMemcpy, cudaMemcpy2D, cudaMemcpyToArray, cudaMemcpy2DToArray,
cudaMemcpy2DFromArray, cudaMemcpyArrayToArray, cudaMemcpy2DArrayToArray,
cudaMemcpyToSymbol, cudaMemcpyFromSymbol, cudaMemcpyAsync, cudaMemcpy2DAsync,
cudaMemcpyToArrayAsync, cudaMemcpy2DToArrayAsync, cudaMemcpyFromArrayAsync,
cudaMemcpy2DFromArrayAsync, cudaMemcpyToSymbolAsync, cudaMemcpyFromSymbolAsync,
cuMemcpyAtoH, cuMemcpyAtoD
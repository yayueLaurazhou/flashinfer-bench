# __host__cudaError_t cudaDeviceGetTexture1DLinearMaxWidth (size_t *maxWidthInElements, const cudaChannelFormatDesc *fmtDesc, int device)

Returns the maximum number of elements allocatable in a 1D linear texture for a given element size.

##### Parameters

**maxWidthInElements**

  - Returns maximum number of texture elements allocatable for given fmtDesc.
**fmtDesc**

  - Texture format description.
**device**

##### Returns

cudaSuccess, cudaErrorUnsupportedLimit, cudaErrorInvalidValue

##### Description

Returns in maxWidthInElements the maximum number of elements allocatable in a 1D linear
texture for given format descriptor fmtDesc.


Note:

**‣** Note that this function may also return error codes from previous, asynchronous launches.


CUDA Runtime API vRelease Version  |  23


Modules





See also:

cuDeviceGetTexture1DLinearMaxWidth
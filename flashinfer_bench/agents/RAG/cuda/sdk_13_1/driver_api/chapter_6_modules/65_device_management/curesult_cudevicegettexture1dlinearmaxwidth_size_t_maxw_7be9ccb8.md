# CUresult cuDeviceGetTexture1DLinearMaxWidth (size_t *maxWidthInElements, CUarray_format format, unsigned numChannels, CUdevice dev)

Returns the maximum number of elements allocatable in a 1D linear texture for a given texture element
size.

###### Parameters

**maxWidthInElements**

  - Returned maximum number of texture elements allocatable for given format and
numChannels.
**format**

  - Texture format.
**numChannels**

  - Number of channels per texture element.
**dev**

  - Device handle.


CUDA Driver API TRM-06703-001 _vRelease Version  |  108


Modules

###### Returns

CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE,
CUDA_ERROR_INVALID_DEVICE

###### Description

Returns in maxWidthInElements the maximum number of texture elements allocatable in a 1D
linear texture for given format and numChannels.


Note:


Note that this function may also return error codes from previous, asynchronous launches.


See also:

cuDeviceGetAttribute, cuDeviceGetCount, cuDeviceGetName, cuDeviceGetUuid, cuDeviceGet,
cudaMemGetInfo, cuDeviceTotalMem
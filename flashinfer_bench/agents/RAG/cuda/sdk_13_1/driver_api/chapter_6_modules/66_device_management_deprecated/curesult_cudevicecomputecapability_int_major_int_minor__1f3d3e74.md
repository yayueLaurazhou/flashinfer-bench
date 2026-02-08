# CUresult cuDeviceComputeCapability (int *major, int *minor, CUdevice dev)

Returns the compute capability of the device.

###### Parameters

**major**

  - Major revision number
**minor**

  - Minor revision number
**dev**

  - Device handle

###### Returns

CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE,
CUDA_ERROR_INVALID_DEVICE

###### Description

Deprecated

This function was deprecated as of CUDA 5.0 and its functionality superseded by
cuDeviceGetAttribute().

Returns in *major and *minor the major and minor revision numbers that define the compute
capability of the device dev.


CUDA Driver API TRM-06703-001 _vRelease Version  |  112


Modules


Note:


Note that this function may also return error codes from previous, asynchronous launches.


See also:

cuDeviceGetAttribute, cuDeviceGetCount, cuDeviceGetName, cuDeviceGetUuid, cuDeviceGet,
cuDeviceTotalMem
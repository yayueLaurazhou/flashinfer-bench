# CUresult cuDeviceTotalMem (size_t *bytes, CUdevice dev)

Returns the total amount of memory on the device.

###### Parameters

**bytes**

  - Returned memory available on device in bytes
**dev**

  - Device handle

###### Returns

CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE,
CUDA_ERROR_INVALID_DEVICE


CUDA Driver API TRM-06703-001 _vRelease Version  |  110


Modules

###### Description

Returns in *bytes the total amount of memory available on the device dev in bytes.


Note:


Note that this function may also return error codes from previous, asynchronous launches.


See also:

cuDeviceGetAttribute, cuDeviceGetCount, cuDeviceGetName, cuDeviceGetUuid, cuDeviceGet,
cuDeviceGetExecAffinitySupport, cudaMemGetInfo
# CUresult cuDeviceGet (CUdevice *device, int ordinal)

Returns a handle to a compute device.

###### Parameters

**device**

  - Returned device handle
**ordinal**

  - Device number to get handle for

###### Returns

CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE,
CUDA_ERROR_INVALID_DEVICE

###### Description

Returns in *device a device handle given an ordinal in the range [0, cuDeviceGetCount()-1].


Note:


Note that this function may also return error codes from previous, asynchronous launches.


See also:

cuDeviceGetAttribute, cuDeviceGetCount, cuDeviceGetName, cuDeviceGetUuid, cuDeviceGetLuid,
cuDeviceTotalMem, cuDeviceGetExecAffinitySupport
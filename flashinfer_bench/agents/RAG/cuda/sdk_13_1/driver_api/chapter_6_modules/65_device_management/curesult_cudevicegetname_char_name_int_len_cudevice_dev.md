# CUresult cuDeviceGetName (char *name, int len, CUdevice dev)

Returns an identifier string for the device.

###### Parameters

**name**

  - Returned identifier string for the device
**len**

  - Maximum length of string to store in name
**dev**

  - Device to get identifier string for

###### Returns

CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE,
CUDA_ERROR_INVALID_DEVICE

###### Description

Returns an ASCII string identifying the device dev in the NULL-terminated string pointed to by
name. len specifies the maximum length of the string that may be returned. name is shortened to the
specified len, if len is less than the device name


CUDA Driver API TRM-06703-001 _vRelease Version  |  106


Modules


Note:


Note that this function may also return error codes from previous, asynchronous launches.


See also:

cuDeviceGetAttribute, cuDeviceGetUuid, cuDeviceGetLuid, cuDeviceGetCount, cuDeviceGet,
cuDeviceTotalMem, cuDeviceGetExecAffinitySupport, cudaGetDeviceProperties
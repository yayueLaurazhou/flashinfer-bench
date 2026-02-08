# CUresult cuDeviceGetUuid (CUuuid *uuid, CUdevice dev)

Return an UUID for the device.

###### Parameters

**uuid**

  - Returned UUID
**dev**

  - Device to get identifier string for

###### Returns

CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
CUDA_ERROR_INVALID_VALUE, CUDA_ERROR_INVALID_DEVICE

###### Description

Returns 16-octets identifying the device dev in the structure pointed by the uuid. If the device is in
MIG mode, returns its MIG UUID which uniquely identifies the subscribed MIG compute instance.


Note:


Note that this function may also return error codes from previous, asynchronous launches.


See also:


CUDA Driver API TRM-06703-001 _vRelease Version  |  109


Modules


cuDeviceGetAttribute, cuDeviceGetCount, cuDeviceGetName, cuDeviceGetLuid, cuDeviceGet,
cuDeviceTotalMem, cudaGetDeviceProperties
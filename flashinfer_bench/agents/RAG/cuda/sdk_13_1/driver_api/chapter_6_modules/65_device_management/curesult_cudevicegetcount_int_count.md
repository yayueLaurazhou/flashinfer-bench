# CUresult cuDeviceGetCount (int *count)

Returns the number of compute-capable devices.

###### Parameters

**count**

  - Returned number of compute-capable devices

###### Returns

CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE

###### Description

Returns in *count the number of devices with compute capability greater than or equal to 2.0 that are
available for execution. If there is no such device, cuDeviceGetCount() returns 0.


Note:


Note that this function may also return error codes from previous, asynchronous launches.


See also:

cuDeviceGetAttribute, cuDeviceGetName, cuDeviceGetUuid, cuDeviceGetLuid, cuDeviceGet,
cuDeviceTotalMem, cuDeviceGetExecAffinitySupport, cudaGetDeviceCount


CUDA Driver API TRM-06703-001 _vRelease Version  |  102


Modules
# CUresult cuDeviceGetExecAffinitySupport (int *pi, CUexecAffinityType type, CUdevice dev)

Returns information about the execution affinity support of the device.

###### Parameters

**pi**

  - 1 if the execution affinity type type is supported by the device, or 0 if not
**type**

  - Execution affinity type to query
**dev**

  - Device handle

###### Returns

CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE,
CUDA_ERROR_INVALID_DEVICE


CUDA Driver API TRM-06703-001 _vRelease Version  |  103


Modules

###### Description

Returns in *pi whether execution affinity type type is supported by device dev. The supported types
are:

CU_EXEC_AFFINITY_TYPE_SM_COUNT: 1 if context with limited SMs is supported by the

###### **‣**

device, or 0 if not;


Note:


Note that this function may also return error codes from previous, asynchronous launches.


See also:

cuDeviceGetAttribute, cuDeviceGetCount, cuDeviceGetName, cuDeviceGetUuid, cuDeviceGet,
cuDeviceTotalMem
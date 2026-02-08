# CUresult cuDeviceGetAttribute (int *pi, CUdevice_attribute attrib, CUdevice dev)

Returns information about the device.

###### Parameters

**pi**

  - Returned device attribute value
**attrib**

  - Device attribute to query
**dev**

  - Device handle


CUDA Driver API TRM-06703-001 _vRelease Version  |  101


Modules

###### Returns

CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE,
CUDA_ERROR_INVALID_DEVICE

###### Description

Returns in *pi the integer value of the attribute attrib on device dev.


Note:


Note that this function may also return error codes from previous, asynchronous launches.


See also:

cuDeviceGetCount, cuDeviceGetName, cuDeviceGetUuid, cuDeviceGet, cuDeviceTotalMem,
cuDeviceGetExecAffinitySupport, cudaDeviceGetAttribute, cudaGetDeviceProperties
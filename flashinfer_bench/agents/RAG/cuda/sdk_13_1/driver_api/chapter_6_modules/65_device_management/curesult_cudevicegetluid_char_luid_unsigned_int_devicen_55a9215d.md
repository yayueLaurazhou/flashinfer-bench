# CUresult cuDeviceGetLuid (char *luid, unsigned int *deviceNodeMask, CUdevice dev)

Return an LUID and device node mask for the device.

###### Parameters

**luid**

  - Returned LUID
**deviceNodeMask**

  - Returned device node mask
**dev**

  - Device to get identifier string for

###### Returns

CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
CUDA_ERROR_INVALID_VALUE, CUDA_ERROR_INVALID_DEVICE

###### Description

Return identifying information (luid and deviceNodeMask) to allow matching device with
graphics APIs.


Note:


Note that this function may also return error codes from previous, asynchronous launches.


See also:


CUDA Driver API TRM-06703-001 _vRelease Version  |  105


Modules


cuDeviceGetAttribute, cuDeviceGetCount, cuDeviceGetName, cuDeviceGet, cuDeviceTotalMem,
cuDeviceGetExecAffinitySupport, cudaGetDeviceProperties
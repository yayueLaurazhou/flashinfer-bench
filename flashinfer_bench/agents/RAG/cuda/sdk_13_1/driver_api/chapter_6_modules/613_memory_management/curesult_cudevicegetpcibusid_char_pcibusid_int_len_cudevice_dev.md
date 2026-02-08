# CUresult cuDeviceGetPCIBusId (char *pciBusId, int len, CUdevice dev)

Returns a PCI Bus Id string for the device.

###### Parameters

**pciBusId**

  - Returned identifier string for the device in the following format [domain]:[bus]:[device].[function]
where domain, bus, device, and function are all hexadecimal values. pciBusId should be
large enough to store 13 characters including the NULL-terminator.
**len**

  - Maximum length of string to store in name
**dev**

  - Device to get identifier string for

###### Returns

CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
CUDA_ERROR_INVALID_VALUE, CUDA_ERROR_INVALID_DEVICE

###### Description

Returns an ASCII string identifying the device dev in the NULL-terminated string pointed to by
pciBusId. len specifies the maximum length of the string that may be returned.


Note:


Note that this function may also return error codes from previous, asynchronous launches.


See also:

cuDeviceGet, cuDeviceGetAttribute, cuDeviceGetByPCIBusId, cudaDeviceGetPCIBusId


CUDA Driver API TRM-06703-001 _vRelease Version  |  188


Modules
# CUresult cuDeviceGetByPCIBusId (CUdevice *dev, const char *pciBusId)

Returns a handle to a compute device.

###### Parameters

**dev**

  - Returned device handle
**pciBusId**

  - String in one of the following forms: [domain]:[bus]:[device].[function] [domain]:[bus]:[device]

[bus]:[device].[function] where domain, bus, device, and function are all hexadecimal
values

###### Returns

CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
CUDA_ERROR_INVALID_VALUE, CUDA_ERROR_INVALID_DEVICE

###### Description

Returns in *device a device handle given a PCI bus ID string.


CUDA Driver API TRM-06703-001 _vRelease Version  |  187


Modules


Note:


Note that this function may also return error codes from previous, asynchronous launches.


See also:

cuDeviceGet, cuDeviceGetAttribute, cuDeviceGetPCIBusId, cudaDeviceGetByPCIBusId
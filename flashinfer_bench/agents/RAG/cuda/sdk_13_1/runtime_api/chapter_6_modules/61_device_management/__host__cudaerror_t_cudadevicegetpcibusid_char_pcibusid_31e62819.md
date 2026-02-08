# __host__cudaError_t cudaDeviceGetPCIBusId (char *pciBusId, int len, int device)

Returns a PCI Bus Id string for the device.

##### Parameters

**pciBusId**

  - Returned identifier string for the device in the following format [domain]:[bus]:[device].[function]
where domain, bus, device, and function are all hexadecimal values. pciBusId should be
large enough to store 13 characters including the NULL-terminator.
**len**

  - Maximum length of string to store in name
**device**

  - Device to get identifier string for

##### Returns

cudaSuccess, cudaErrorInvalidValue, cudaErrorInvalidDevice


CUDA Runtime API vRelease Version  |  21


Modules

##### Description

Returns an ASCII string identifying the device dev in the NULL-terminated string pointed to by
pciBusId. len specifies the maximum length of the string that may be returned.



See also:

cudaDeviceGetByPCIBusId, cuDeviceGetPCIBusId
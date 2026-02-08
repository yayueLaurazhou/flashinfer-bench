# __host__cudaError_t cudaDeviceGetByPCIBusId (int *device, const char *pciBusId)

Returns a handle to a compute device.

##### Parameters

**device**

  - Returned device ordinal
**pciBusId**

  - String in one of the following forms: [domain]:[bus]:[device].[function] [domain]:[bus]:[device]

[bus]:[device].[function] where domain, bus, device, and function are all hexadecimal
values

##### Returns

cudaSuccess, cudaErrorInvalidValue, cudaErrorInvalidDevice

##### Description

Returns in *device a device ordinal given a PCI bus ID string.





See also:

cudaDeviceGetPCIBusId, cuDeviceGetByPCIBusId
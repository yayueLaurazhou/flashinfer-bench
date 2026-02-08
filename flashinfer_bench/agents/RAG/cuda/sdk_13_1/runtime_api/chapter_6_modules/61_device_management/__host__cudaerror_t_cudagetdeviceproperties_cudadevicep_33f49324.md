# __host__cudaError_t cudaGetDeviceProperties (cudaDeviceProp *prop, int device)

Returns information about the compute-device.

##### Parameters

**prop**

  - Properties for the specified device
**device**

  - Device number to get properties for

##### Returns

cudaSuccess, cudaErrorInvalidDevice

##### Description

Returns in *prop the properties of device dev.


Note:

**‣** Note that this function may also return error codes from previous, asynchronous launches.


CUDA Runtime API vRelease Version  |  33


Modules





See also:

cudaGetDeviceCount, cudaGetDevice, cudaSetDevice, cudaChooseDevice, cudaDeviceGetAttribute,
cudaInitDevice, cuDeviceGetAttribute, cuDeviceGetName
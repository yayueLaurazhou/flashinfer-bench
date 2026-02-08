# __host____device__cudaError_t cudaGetDeviceCount (int *count)

Returns the number of compute-capable devices.

##### Parameters

**count**

  - Returns the number of devices with compute capability greater or equal to 2.0

##### Returns

cudaSuccess


CUDA Runtime API vRelease Version  |  31


Modules

##### Description

Returns in *count the number of devices with compute capability greater or equal to 2.0 that are
available for execution.



See also:

cudaGetDevice, cudaSetDevice, cudaGetDeviceProperties, cudaChooseDevice, cudaInitDevice,
cuDeviceGetCount
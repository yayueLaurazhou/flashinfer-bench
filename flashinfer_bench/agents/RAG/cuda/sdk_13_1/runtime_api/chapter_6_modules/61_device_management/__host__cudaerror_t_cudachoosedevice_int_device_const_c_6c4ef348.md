# __host__cudaError_t cudaChooseDevice (int *device, const cudaDeviceProp *prop)

Select compute-device which best matches criteria.

##### Parameters

**device**

  - Device with best match
**prop**

  - Desired device properties

##### Returns

cudaSuccess, cudaErrorInvalidValue

##### Description

Returns in *device the device which has properties that best match *prop.





CUDA Runtime API vRelease Version  |  10


Modules


See also:

cudaGetDeviceCount, cudaGetDevice, cudaSetDevice, cudaGetDeviceProperties, cudaInitDevice
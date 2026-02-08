# __host____device__cudaError_t cudaDeviceGetAttribute (int *value, cudaDeviceAttr attr, int device)

Returns information about the device.

##### Parameters

**value**

  - Returned device attribute value
**attr**

  - Device attribute to query
**device**

  - Device number to query

##### Returns

cudaSuccess, cudaErrorInvalidDevice, cudaErrorInvalidValue

##### Description

Returns in *value the integer value of the attribute attr on device device.



See also:

cudaGetDeviceCount, cudaGetDevice, cudaSetDevice, cudaChooseDevice, cudaGetDeviceProperties,
cudaInitDevice, cuDeviceGetAttribute


CUDA Runtime API vRelease Version  |  12


Modules
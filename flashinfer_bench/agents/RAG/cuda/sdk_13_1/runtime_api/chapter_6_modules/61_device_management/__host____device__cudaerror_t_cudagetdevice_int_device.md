# __host____device__cudaError_t cudaGetDevice (int *device)

Returns which device is currently being used.

##### Parameters

**device**

  - Returns the device on which the active host thread executes the device code.

##### Returns

cudaSuccess, cudaErrorInvalidValue, cudaErrorDeviceUnavailable,

##### Description

Returns in *device the current device for the calling host thread.



See also:

cudaGetDeviceCount, cudaSetDevice, cudaGetDeviceProperties, cudaChooseDevice, cuCtxGetCurrent
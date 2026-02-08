# __host__cudaError_t cudaStreamGetDevice (cudaStream_t hStream, int *device)

Query the device of a stream.

##### Parameters

**hStream**

  - Handle to the stream to be queried
**device**

  - Returns the device to which the stream belongs

##### Returns

cudaSuccess, cudaErrorInvalidValue, cudaErrorDeviceUnavailable,

##### Description

Returns in *device the device of the stream.











See also:

cudaSetDevice, cudaGetDevice, cudaStreamCreate, cudaStreamGetPriority, cudaStreamGetFlags,
cuStreamGetId
# __host__cudaError_t cudaSetValidDevices (int *device_arr, int len)

Set a list of devices that can be used for CUDA.

##### Parameters

**device_arr**

  - List of devices to try
**len**

  - Number of devices in specified list

##### Returns

cudaSuccess, cudaErrorInvalidValue, cudaErrorInvalidDevice

##### Description

Sets a list of devices for CUDA execution in priority order using device_arr. The parameter len
specifies the number of elements in the list. CUDA will try devices from the list sequentially until it
finds one that works. If this function is not called, or if it is called with a len of 0, then CUDA will
go back to its default behavior of trying devices sequentially from a default list containing all of the
available CUDA devices in the system. If a specified device ID in the list does not exist, this function
will return cudaErrorInvalidDevice. If len is not 0 and device_arr is NULL or if len exceeds the
number of devices in the system, then cudaErrorInvalidValue is returned.





See also:

cudaGetDeviceCount, cudaSetDevice, cudaGetDeviceProperties, cudaSetDeviceFlags,
cudaChooseDevice
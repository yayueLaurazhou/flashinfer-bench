# CUresult cuDeviceSetMemPool (CUdevice dev, CUmemoryPool pool)

Sets the current memory pool of a device.

###### Returns

CUDA_SUCCESS, CUDA_ERROR_INVALID_VALUE

###### Description

The memory pool must be local to the specified device. cuMemAllocAsync allocates from the current
mempool of the provided stream's device. By default, a device's current memory pool is its default
memory pool.





See also:

cuDeviceGetDefaultMemPool, cuDeviceGetMemPool, cuMemPoolCreate, cuMemPoolDestroy,
cuMemAllocFromPoolAsync
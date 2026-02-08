# CUresult cuDeviceGetMemPool (CUmemoryPool *pool, CUdevice dev)

Gets the current mempool for a device.

###### Returns

CUDA_SUCCESS, CUDA_ERROR_INVALID_VALUE

###### Description

Returns the last pool provided to cuDeviceSetMemPool for this device or the device's default memory
pool if cuDeviceSetMemPool has never been called. By default the current mempool is the default
mempool for a device. Otherwise the returned pool must have been set with cuDeviceSetMemPool.


See also:

cuDeviceGetDefaultMemPool, cuMemPoolCreate, cuDeviceSetMemPool
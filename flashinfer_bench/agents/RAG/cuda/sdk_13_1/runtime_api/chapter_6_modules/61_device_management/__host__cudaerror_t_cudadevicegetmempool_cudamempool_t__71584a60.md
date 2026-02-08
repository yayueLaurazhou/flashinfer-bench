# __host__cudaError_t cudaDeviceGetMemPool (cudaMemPool_t *memPool, int device)

Gets the current mempool for a device.

##### Returns

cudaSuccess, cudaErrorInvalidValue cudaErrorNotSupported

##### Description

Returns the last pool provided to cudaDeviceSetMemPool for this device or the device's default
memory pool if cudaDeviceSetMemPool has never been called. By default the current mempool
is the default mempool for a device, otherwise the returned pool must have been set with
cuDeviceSetMemPool or cudaDeviceSetMemPool.







See also:


CUDA Runtime API vRelease Version  |  17


Modules


cuDeviceGetMemPool, cudaDeviceGetDefaultMemPool, cudaDeviceSetMemPool
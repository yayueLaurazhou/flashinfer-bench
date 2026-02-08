# __host__cudaError_t cudaDeviceGetDefaultMemPool (cudaMemPool_t *memPool, int device)

Returns the default mempool of a device.

##### Returns

cudaSuccess, cudaErrorInvalidDevice, cudaErrorInvalidValue cudaErrorNotSupported


CUDA Runtime API vRelease Version  |  14


Modules

##### Description

The default mempool of a device contains device memory from that device.





See also:

cuDeviceGetDefaultMemPool, cudaMallocAsync, cudaMemPoolTrimTo, cudaMemPoolGetAttribute,
cudaDeviceSetMemPool, cudaMemPoolSetAttribute, cudaMemPoolSetAccess
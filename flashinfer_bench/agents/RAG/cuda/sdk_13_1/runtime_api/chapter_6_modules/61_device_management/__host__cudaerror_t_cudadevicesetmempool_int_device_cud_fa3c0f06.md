# __host__cudaError_t cudaDeviceSetMemPool (int device, cudaMemPool_t memPool)

Sets the current memory pool of a device.

##### Returns

cudaSuccess, cudaErrorInvalidValue cudaErrorInvalidDevice cudaErrorNotSupported

##### Description

The memory pool must be local to the specified device. Unless a mempool is specified in the
cudaMallocAsync call, cudaMallocAsync allocates from the current mempool of the provided stream's
device. By default, a device's current memory pool is its default memory pool.







See also:

cuDeviceSetMemPool, cudaDeviceGetMemPool, cudaDeviceGetDefaultMemPool,
cudaMemPoolCreate, cudaMemPoolDestroy, cudaMallocFromPoolAsync
# __host__cudaError_t cudaMemPoolDestroy (cudaMemPool_t memPool)

Destroys the specified memory pool.

##### Returns

cudaSuccess, cudaErrorInvalidValue

##### Description

If any pointers obtained from this pool haven't been freed or the pool has free operations that haven't
completed when cudaMemPoolDestroy is invoked, the function will return immediately and the
resources associated with the pool will be released automatically once there are no more outstanding
allocations.


CUDA Runtime API vRelease Version  |  216


Modules


Destroying the current mempool of a device sets the default mempool of that device as the current
mempool for that device.


Note:


A device's default memory pool cannot be destroyed.


See also:

cuMemPoolDestroy, cudaFreeAsync, cudaDeviceSetMemPool, cudaDeviceGetDefaultMemPool,
cudaDeviceGetMemPool, cudaMemPoolCreate
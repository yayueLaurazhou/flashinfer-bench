# __host__cudaError_t cudaMemGetMemPool (cudaMemPool_t *memPool, cudaMemLocation *location, cudaMemAllocationType type)

Gets the current memory pool for a given memory location and allocation type.

##### Returns

cudaSuccess, cudaErrorInvalidValue

##### Description

The memory location can be of one of cudaMemLocationTypeDevice, cudaMemLocationTypeHost or
cudaMemLocationTypeHostNuma. The allocation type can be one of cudaMemAllocationTypePinned
or cudaMemAllocationTypeManaged. When the allocation type is cudaMemAllocationTypeManaged,
the location type can also be cudaMemLocationTypeNone to indicate no preferred location for the
managed memory pool. In all other cases, the call return cudaErrorInvalidValue


CUDA Runtime API vRelease Version  |  214


Modules


Returns the last pool provided to cudaMemSetMemPool or cudaDeviceSetMemPool for this
location and allocation type or the location's default memory pool if cudaMemSetMemPool
or cudaDeviceSetMemPool for that allocType and location has never been called. By default
the current mempool of a location is the default mempool for a device that can be obtained
via cudaMemGetDefaultMemPool Otherwise the returned pool must have been set with
cudaDeviceSetMemPool.


See also:

cuDeviceGetDefaultMemPool, cuMemPoolCreate, cuDeviceSetMemPool, cuMemSetMemPool
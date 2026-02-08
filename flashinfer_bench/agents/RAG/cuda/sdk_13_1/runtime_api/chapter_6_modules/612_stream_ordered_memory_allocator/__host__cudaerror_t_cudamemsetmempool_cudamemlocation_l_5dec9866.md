# __host__cudaError_t cudaMemSetMemPool (cudaMemLocation *location, cudaMemAllocationType type, cudaMemPool_t memPool)

Sets the current memory pool for a memory location and allocation type.

##### Returns

cudaSuccess, cudaErrorInvalidValue

##### Description

The memory location can be of one of cudaMemLocationTypeDevice, cudaMemLocationTypeHost or
cudaMemLocationTypeHostNuma. The allocation type can be one of cudaMemAllocationTypePinned
or cudaMemAllocationTypeManaged. When the allocation type is cudaMemAllocationTypeManaged,
the location type can also be cudaMemLocationTypeNone to indicate no preferred location for the
managed memory pool. In all other cases, the call return cudaErrorInvalidValue

When a memory pool is set as the current memory pool, the location parameter should be
the same as the location of the pool. If the location type or index don't match, the call returns
cudaErrorInvalidValue. The type of memory pool should also match the parameter allocType. Else
the call returns cudaErrorInvalidValue. By default, a memory location's current memory pool is its
default memory pool. If the location type is cudaMemLocationTypeDevice and the allocation type is
cudaMemAllocationTypePinned, then this API is the equivalent of calling cudaDeviceSetMemPool
with the location id as the device. For further details on the implications, please refer to the
documentation for cudaDeviceSetMemPool.





See also:


CUDA Runtime API vRelease Version  |  225


Modules


cuDeviceGetDefaultMemPool, cuDeviceGetMemPool, cuMemGetMemPool, cuMemPoolCreate,
cuMemPoolDestroy, cuMemAllocFromPoolAsync
# __host__cudaError_t cudaMemGetDefaultMemPool (cudaMemPool_t *memPool, cudaMemLocation *location, cudaMemAllocationType type)

Returns the default memory pool for a given location and allocation type.

##### Returns

cudaSuccess, cudaErrorInvalidValue, cudaErrorNotSupported,

##### Description

The memory location can be of one of cudaMemLocationTypeDevice, cudaMemLocationTypeHost or
cudaMemLocationTypeHostNuma. The allocation type can be one of cudaMemAllocationTypePinned
or cudaMemAllocationTypeManaged. When the allocation type is cudaMemAllocationTypeManaged,
the location type can also be cudaMemLocationTypeNone to indicate no preferred location for the
managed memory pool. In all other cases, the call return cudaErrorInvalidValue


See also:

cuMemAllocAsync, cuMemPoolTrimTo, cuMemPoolGetAttribute, cuMemPoolSetAttribute,
cuMemPoolSetAccess, cuMemGetMemPool, cuMemPoolCreate
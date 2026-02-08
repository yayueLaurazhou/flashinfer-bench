# CUresult cuMemGetMemPool (CUmemoryPool *pool, CUmemLocation *location, CUmemAllocationType type)

Gets the current memory pool for a memory location and of a particular allocation type.

###### Returns

CUDA_SUCCESS, CUDA_ERROR_INVALID_VALUE

###### Description

The memory location can be of one of CU_MEM_LOCATION_TYPE_DEVICE,
CU_MEM_LOCATION_TYPE_HOST or CU_MEM_LOCATION_TYPE_HOST_NUMA.
The allocation type can be one of CU_MEM_ALLOCATION_TYPE_PINNED
or CU_MEM_ALLOCATION_TYPE_MANAGED. When the allocation type is
CU_MEM_ALLOCATION_TYPE_MANAGED, the location type can also be
CU_MEM_LOCATION_TYPE_NONE to indicate no preferred location for the managed memory
pool. In all other cases, the call returns CUDA_ERROR_INVALID_VALUE

Returns the last pool provided to cuMemSetMemPool or cuDeviceSetMemPool for this location and
allocation type or the location's default memory pool if cuMemSetMemPool or cuDeviceSetMemPool
for that allocType and location has never been called. By default the current mempool of a
location is the default mempool for a device. Otherwise the returned pool must have been set with
cuDeviceSetMemPool.


See also:

cuDeviceGetDefaultMemPool, cuMemPoolCreate, cuDeviceSetMemPool, cuMemSetMemPool


CUDA Driver API TRM-06703-001 _vRelease Version  |  290


Modules
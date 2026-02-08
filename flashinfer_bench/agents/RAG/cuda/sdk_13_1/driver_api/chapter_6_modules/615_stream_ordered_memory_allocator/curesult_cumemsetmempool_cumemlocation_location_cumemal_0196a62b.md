# CUresult cuMemSetMemPool (CUmemLocation *location, CUmemAllocationType type, CUmemoryPool pool)

Sets the current memory pool for a memory location and allocation type.

###### Returns

CUDA_SUCCESS, CUDA_ERROR_INVALID_VALUE

###### Description

The memory location can be of one of CU_MEM_LOCATION_TYPE_DEVICE,
CU_MEM_LOCATION_TYPE_HOST or CU_MEM_LOCATION_TYPE_HOST_NUMA.
The allocation type can be one of CU_MEM_ALLOCATION_TYPE_PINNED
or CU_MEM_ALLOCATION_TYPE_MANAGED. When the allocation type is
CU_MEM_ALLOCATION_TYPE_MANAGED, the location type can also be
CU_MEM_LOCATION_TYPE_NONE to indicate no preferred location for the managed memory
pool. In all other cases, the call returns CUDA_ERROR_INVALID_VALUE.

When a memory pool is set as the current memory pool, the location parameter should be
the same as the location of the pool. The location and allocation type specified must match


CUDA Driver API TRM-06703-001 _vRelease Version  |  300


Modules


those of the pool otherwise CUDA_ERROR_INVALID_VALUE is returned. By default, a
memory location's current memory pool is its default memory pool that can be obtained via
cuMemGetDefaultMemPool. If the location type is CU_MEM_LOCATION_TYPE_DEVICE and
the allocation type is CU_MEM_ALLOCATION_TYPE_PINNED, then this API is the equivalent of
calling cuDeviceSetMemPool with the location id as the device. For further details on the implications,
please refer to the documentation for cuDeviceSetMemPool.





See also:

cuDeviceGetDefaultMemPool, cuDeviceGetMemPool, cuMemGetMemPool, cuMemPoolCreate,
cuMemPoolDestroy, cuMemAllocFromPoolAsync
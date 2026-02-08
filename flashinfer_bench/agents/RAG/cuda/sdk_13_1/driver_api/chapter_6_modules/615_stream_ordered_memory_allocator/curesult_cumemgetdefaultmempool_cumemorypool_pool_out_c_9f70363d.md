# CUresult cuMemGetDefaultMemPool (CUmemoryPool *pool_out, CUmemLocation *location, CUmemAllocationType type)

Returns the default memory pool for a given location and allocation type.

###### Returns

CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED CUDA_ERROR_NOT_INITIALIZED,
CUDA_ERROR_INVALID_VALUE, CUDA_ERROR_INVALID_DEVICE,
CUDA_ERROR_NOT_SUPPORTED

###### Description

The memory location can be of one of CU_MEM_LOCATION_TYPE_DEVICE,
CU_MEM_LOCATION_TYPE_HOST or CU_MEM_LOCATION_TYPE_HOST_NUMA.


CUDA Driver API TRM-06703-001 _vRelease Version  |  289


Modules


The allocation type can be one of CU_MEM_ALLOCATION_TYPE_PINNED
or CU_MEM_ALLOCATION_TYPE_MANAGED. When the allocation type is
CU_MEM_ALLOCATION_TYPE_MANAGED, the location type can also be
CU_MEM_LOCATION_TYPE_NONE to indicate no preferred location for the managed memory
pool. In all other cases, the call returns CUDA_ERROR_INVALID_VALUE.


Note:


Note that this function may also return error codes from previous, asynchronous launches.


See also:

cuMemAllocAsync, cuMemPoolTrimTo, cuMemPoolGetAttribute, cuMemPoolSetAttribute,
cuMemPoolSetAccess, cuMemGetMemPool, cuMemPoolCreate
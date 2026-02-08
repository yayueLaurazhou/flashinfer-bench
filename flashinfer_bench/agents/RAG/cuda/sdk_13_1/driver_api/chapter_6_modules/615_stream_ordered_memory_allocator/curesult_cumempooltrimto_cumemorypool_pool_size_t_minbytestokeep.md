# CUresult cuMemPoolTrimTo (CUmemoryPool pool, size_t minBytesToKeep)

Tries to release memory back to the OS.

###### Parameters

**pool**

  - The memory pool to trim


CUDA Driver API TRM-06703-001 _vRelease Version  |  299


Modules


**minBytesToKeep**

  - If the pool has less than minBytesToKeep reserved, the TrimTo operation is a no-op. Otherwise
the pool will be guaranteed to have at least minBytesToKeep bytes reserved after the operation.

###### Returns

CUDA_SUCCESS, CUDA_ERROR_NOT_INITIALIZED, CUDA_ERROR_INVALID_VALUE

###### Description

Releases memory back to the OS until the pool contains fewer than minBytesToKeep reserved bytes,
or there is no more memory that the allocator can safely release. The allocator cannot release OS
allocations that back outstanding asynchronous allocations. The OS allocations may happen at different
granularity from the user allocations.


Note:

**‣** : Allocations that have not been freed count as outstanding.

**‣** : Allocations that have been asynchronously freed but whose completion has not been observed on
the host (eg. by a synchronize) can count as outstanding.


See also:

cuMemAllocAsync, cuMemFreeAsync, cuDeviceGetDefaultMemPool, cuDeviceGetMemPool,
cuMemPoolCreate
# __host__cudaError_t cudaMemPoolTrimTo (cudaMemPool_t memPool, size_t minBytesToKeep)

Tries to release memory back to the OS.

##### Parameters

**memPool**
**minBytesToKeep**

  - If the pool has less than minBytesToKeep reserved, the TrimTo operation is a no-op. Otherwise
the pool will be guaranteed to have at least minBytesToKeep bytes reserved after the operation.

##### Returns

cudaSuccess, cudaErrorInvalidValue

##### Description

Releases memory back to the OS until the pool contains fewer than minBytesToKeep reserved bytes,
or there is no more memory that the allocator can safely release. The allocator cannot release OS
allocations that back outstanding asynchronous allocations. The OS allocations may happen at different
granularity from the user allocations.


Note:

**‣** : Allocations that have not been freed count as outstanding.

**‣** : Allocations that have been asynchronously freed but whose completion has not been observed on
the host (eg. by a synchronize) can count as outstanding.


CUDA Runtime API vRelease Version  |  224


Modules





See also:

cuMemPoolTrimTo, cudaMallocAsync, cudaFreeAsync, cudaDeviceGetDefaultMemPool,
cudaDeviceGetMemPool, cudaMemPoolCreate
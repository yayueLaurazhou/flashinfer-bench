# __host__cudaError_t cudaMemPoolSetAccess (cudaMemPool_t memPool, const cudaMemAccessDesc *descList, size_t count)

Controls visibility of pools between devices.

##### Parameters

**memPool**
**descList**
**count**

  - Number of descriptors in the map array.


CUDA Runtime API vRelease Version  |  222


Modules

##### Returns

cudaSuccess, cudaErrorInvalidValue

##### Description

See also:

cuMemPoolSetAccess, cudaMemPoolGetAccess, cudaMallocAsync, cudaFreeAsync
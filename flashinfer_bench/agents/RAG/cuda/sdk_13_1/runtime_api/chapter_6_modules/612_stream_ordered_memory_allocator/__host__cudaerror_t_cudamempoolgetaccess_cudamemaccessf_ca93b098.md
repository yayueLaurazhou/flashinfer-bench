# __host__cudaError_t cudaMemPoolGetAccess (cudaMemAccessFlags *flags, cudaMemPool_t memPool, cudaMemLocation *location)

Returns the accessibility of a pool from a device.

##### Parameters

**flags**

  - the accessibility of the pool from the specified location
**memPool**

  - the pool being queried
**location**

  - the location accessing the pool

##### Description

Returns the accessibility of the pool's memory from the specified location.


See also:

cuMemPoolGetAccess, cudaMemPoolSetAccess
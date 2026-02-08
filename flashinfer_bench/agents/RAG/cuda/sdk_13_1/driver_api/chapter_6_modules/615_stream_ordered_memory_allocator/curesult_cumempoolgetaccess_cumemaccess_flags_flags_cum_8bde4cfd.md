# CUresult cuMemPoolGetAccess (CUmemAccess_flags *flags, CUmemoryPool memPool, CUmemLocation *location)

Returns the accessibility of a pool from a device.

###### Parameters

**flags**

  - the accessibility of the pool from the specified location
**memPool**

  - the pool being queried
**location**

  - the location accessing the pool


CUDA Driver API TRM-06703-001 _vRelease Version  |  294


Modules

###### Description

Returns the accessibility of the pool's memory from the specified location.


See also:

cuMemAllocAsync, cuMemFreeAsync, cuDeviceGetDefaultMemPool, cuDeviceGetMemPool,
cuMemPoolCreate
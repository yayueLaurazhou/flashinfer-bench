# CUresult cuMemPoolDestroy (CUmemoryPool pool)

Destroys the specified memory pool.

###### Returns

CUDA_SUCCESS, CUDA_ERROR_INVALID_VALUE

###### Description

If any pointers obtained from this pool haven't been freed or the pool has free operations that haven't
completed when cuMemPoolDestroy is invoked, the function will return immediately and the resources
associated with the pool will be released automatically once there are no more outstanding allocations.

Destroying the current mempool of a device sets the default mempool of that device as the current
mempool for that device.


Note:


A device's default memory pool cannot be destroyed.


See also:


CUDA Driver API TRM-06703-001 _vRelease Version  |  292


Modules


cuMemFreeAsync, cuDeviceSetMemPool, cuDeviceGetMemPool, cuDeviceGetDefaultMemPool,
cuMemPoolCreate
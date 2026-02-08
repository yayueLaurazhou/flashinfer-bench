# CUresult cuMemRetainAllocationHandle (CUmemGenericAllocationHandle *handle, void *addr)

Given an address addr, returns the allocation handle of the backing memory allocation.

###### Parameters

**handle**
CUDA Memory handle for the backing memory allocation.
**addr**
Memory address to query, that has been mapped previously.


CUDA Driver API TRM-06703-001 _vRelease Version  |  283


Modules

###### Returns

CUDA_SUCCESS, CUDA_ERROR_INVALID_VALUE, CUDA_ERROR_NOT_INITIALIZED,
CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_PERMITTED,
CUDA_ERROR_NOT_SUPPORTED

###### Description

The handle is guaranteed to be the same handle value used to map the memory. If the address requested
is not mapped, the function will fail. The returned handle must be released with corresponding number
of calls to cuMemRelease.





See also:

cuMemCreate, cuMemRelease, cuMemMap
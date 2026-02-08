# CUresult cuMemPoolSetAccess (CUmemoryPool pool, const CUmemAccessDesc *map, size_t count)

Controls visibility of pools between devices.

###### Parameters

**pool**

  - The pool being modified
**map**

  - Array of access descriptors. Each descriptor instructs the access to enable for a single gpu.
**count**

  - Number of descriptors in the map array.

###### Returns

CUDA_SUCCESS, CUDA_ERROR_NOT_INITIALIZED, CUDA_ERROR_INVALID_VALUE

###### Description

See also:

cuMemAllocAsync, cuMemFreeAsync, cuDeviceGetDefaultMemPool, cuDeviceGetMemPool,
cuMemPoolCreate
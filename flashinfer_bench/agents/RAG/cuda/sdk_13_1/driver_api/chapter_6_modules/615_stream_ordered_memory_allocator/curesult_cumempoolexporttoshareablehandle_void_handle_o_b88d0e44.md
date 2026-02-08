# CUresult cuMemPoolExportToShareableHandle (void *handle_out, CUmemoryPool pool, CUmemAllocationHandleType handleType, unsigned long long flags)

Exports a memory pool to the requested handle type.

###### Parameters

**handle_out**

  - Returned OS handle


CUDA Driver API TRM-06703-001 _vRelease Version  |  293


Modules


**pool**

  - pool to export
**handleType**

  - the type of handle to create
**flags**

  - must be 0

###### Returns

CUDA_SUCCESS, CUDA_ERROR_INVALID_VALUE, CUDA_ERROR_NOT_INITIALIZED,
CUDA_ERROR_OUT_OF_MEMORY

###### Description

Given an IPC capable mempool, create an OS handle to share the pool with another
process. A recipient process can convert the shareable handle into a mempool with
cuMemPoolImportFromShareableHandle. Individual pointers can then be shared with the
cuMemPoolExportPointer and cuMemPoolImportPointer APIs. The implementation of what the
shareable handle is and how it can be transferred is defined by the requested handle type.


Note:


: To create an IPC capable mempool, create a mempool with a CUmemAllocationHandleType other than
CU_MEM_HANDLE_TYPE_NONE.


See also:

cuMemPoolImportFromShareableHandle, cuMemPoolExportPointer, cuMemPoolImportPointer,
cuMemAllocAsync, cuMemFreeAsync, cuDeviceGetDefaultMemPool, cuDeviceGetMemPool,
cuMemPoolCreate, cuMemPoolSetAccess, cuMemPoolSetAttribute
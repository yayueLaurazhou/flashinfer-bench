# CUresult cuMemPoolImportFromShareableHandle (CUmemoryPool *pool_out, void *handle, CUmemAllocationHandleType handleType, unsigned long long flags)

imports a memory pool from a shared handle.

###### Parameters

**pool_out**

  - Returned memory pool
**handle**

  - OS handle of the pool to open
**handleType**

  - The type of handle being imported
**flags**

  - must be 0

###### Returns

CUDA_SUCCESS, CUDA_ERROR_INVALID_VALUE, CUDA_ERROR_NOT_INITIALIZED,
CUDA_ERROR_OUT_OF_MEMORY

###### Description

Specific allocations can be imported from the imported pool with cuMemPoolImportPointer.


CUDA Driver API TRM-06703-001 _vRelease Version  |  296


Modules



If handleType is CU_MEM_HANDLE_TYPE_FABRIC and the importer process has not
been granted access to the same IMEX channel as the exporter process, this API will error as
CUDA_ERROR_NOT_PERMITTED.





See also:

cuMemPoolExportToShareableHandle, cuMemPoolExportPointer, cuMemPoolImportPointer
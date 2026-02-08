# CUresult cuMemPoolExportPointer (CUmemPoolPtrExportData *shareData_out, CUdeviceptr ptr)

Export data to share a memory pool allocation between processes.

###### Parameters

**shareData_out**

  - Returned export data
**ptr**

  - pointer to memory being exported

###### Returns

CUDA_SUCCESS, CUDA_ERROR_INVALID_VALUE, CUDA_ERROR_NOT_INITIALIZED,
CUDA_ERROR_OUT_OF_MEMORY

###### Description

Constructs shareData_out for sharing a specific allocation from an already shared memory pool.
The recipient process can import the allocation with the cuMemPoolImportPointer api. The data is not
a handle and may be shared through any IPC mechanism.


See also:

cuMemPoolExportToShareableHandle, cuMemPoolImportFromShareableHandle,
cuMemPoolImportPointer
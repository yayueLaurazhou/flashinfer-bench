# __host__cudaError_t cudaMemPoolExportPointer (cudaMemPoolPtrExportData *exportData, void *ptr)

Export data to share a memory pool allocation between processes.

##### Parameters

**exportData**
**ptr**

  - pointer to memory being exported

##### Returns

cudaSuccess, cudaErrorInvalidValue, cudaErrorOutOfMemory

##### Description

Constructs shareData_out for sharing a specific allocation from an already shared memory pool.
The recipient process can import the allocation with the cudaMemPoolImportPointer api. The data is
not a handle and may be shared through any IPC mechanism.


See also:

cuMemPoolExportPointer, cudaMemPoolExportToShareableHandle,
cudaMemPoolImportFromShareableHandle, cudaMemPoolImportPointer


CUDA Runtime API vRelease Version  |  217


Modules
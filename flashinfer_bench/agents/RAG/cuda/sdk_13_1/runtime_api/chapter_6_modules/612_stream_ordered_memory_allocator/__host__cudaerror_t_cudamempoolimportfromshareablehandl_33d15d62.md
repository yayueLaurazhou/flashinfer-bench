# __host__cudaError_t cudaMemPoolImportFromShareableHandle (cudaMemPool_t *memPool, void *shareableHandle, cudaMemAllocationHandleType handleType, unsigned int flags)

imports a memory pool from a shared handle.

##### Parameters

**memPool**
**shareableHandle**
**handleType**

  - The type of handle being imported
**flags**

  - must be 0

##### Returns

cudaSuccess, cudaErrorInvalidValue, cudaErrorOutOfMemory

##### Description

Specific allocations can be imported from the imported pool with cudaMemPoolImportPointer.





See also:

cuMemPoolImportFromShareableHandle, cudaMemPoolExportToShareableHandle,
cudaMemPoolExportPointer, cudaMemPoolImportPointer


CUDA Runtime API vRelease Version  |  221


Modules
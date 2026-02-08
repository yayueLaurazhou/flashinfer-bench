# __host__cudaError_t cudaMemPoolExportToShareableHandle (void *shareableHandle, cudaMemPool_t memPool, cudaMemAllocationHandleType handleType, unsigned int flags)

Exports a memory pool to the requested handle type.

##### Parameters

**shareableHandle**
**memPool**
**handleType**

  - the type of handle to create
**flags**

  - must be 0

##### Returns

cudaSuccess, cudaErrorInvalidValue, cudaErrorOutOfMemory

##### Description

Given an IPC capable mempool, create an OS handle to share the pool with another
process. A recipient process can convert the shareable handle into a mempool with
cudaMemPoolImportFromShareableHandle. Individual pointers can then be shared with the
cudaMemPoolExportPointer and cudaMemPoolImportPointer APIs. The implementation of what the
shareable handle is and how it can be transferred is defined by the requested handle type.


Note:


: To create an IPC capable mempool, create a mempool with a CUmemAllocationHandleType other than
cudaMemHandleTypeNone.


See also:

cuMemPoolExportToShareableHandle, cudaMemPoolImportFromShareableHandle,
cudaMemPoolExportPointer, cudaMemPoolImportPointer


CUDA Runtime API vRelease Version  |  218


Modules
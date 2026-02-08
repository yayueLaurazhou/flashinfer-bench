# __host__cudaError_t cudaMemPoolImportPointer (void **ptr, cudaMemPool_t memPool, cudaMemPoolPtrExportData *exportData)

Import a memory pool allocation from another process.

##### Returns

CUDA_SUCCESS, CUDA_ERROR_INVALID_VALUE, CUDA_ERROR_NOT_INITIALIZED,
CUDA_ERROR_OUT_OF_MEMORY

##### Description

Returns in ptr_out a pointer to the imported memory. The imported memory must not be accessed
before the allocation operation completes in the exporting process. The imported memory must be
freed from all importing processes before being freed in the exporting process. The pointer may be
freed with cudaFree or cudaFreeAsync. If cudaFreeAsync is used, the free must be completed on the
importing process before the free operation on the exporting process.





See also:

cuMemPoolImportPointer, cudaMemPoolExportToShareableHandle,
cudaMemPoolImportFromShareableHandle, cudaMemPoolExportPointer
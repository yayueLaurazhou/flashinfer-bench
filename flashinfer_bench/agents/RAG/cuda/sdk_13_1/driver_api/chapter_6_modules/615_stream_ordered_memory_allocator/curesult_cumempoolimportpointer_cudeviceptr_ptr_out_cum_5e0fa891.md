# CUresult cuMemPoolImportPointer (CUdeviceptr *ptr_out, CUmemoryPool pool, CUmemPoolPtrExportData *shareData)

Import a memory pool allocation from another process.

###### Parameters

**ptr_out**

  - pointer to imported memory
**pool**

  - pool from which to import
**shareData**

  - data specifying the memory to import

###### Returns

CUDA_SUCCESS, CUDA_ERROR_INVALID_VALUE, CUDA_ERROR_NOT_INITIALIZED,
CUDA_ERROR_OUT_OF_MEMORY

###### Description

Returns in ptr_out a pointer to the imported memory. The imported memory must not be accessed
before the allocation operation completes in the exporting process. The imported memory must be
freed from all importing processes before being freed in the exporting process. The pointer may be
freed with cuMemFree or cuMemFreeAsync. If cuMemFreeAsync is used, the free must be completed
on the importing process before the free operation on the exporting process.


Note:


CUDA Driver API TRM-06703-001 _vRelease Version  |  297


Modules


The cuMemFreeAsync api may be used in the exporting process before the cuMemFreeAsync operation
completes in its stream as long as the cuMemFreeAsync in the exporting process specifies a stream with
a stream dependency on the importing process's cuMemFreeAsync.


See also:

cuMemPoolExportToShareableHandle, cuMemPoolImportFromShareableHandle,
cuMemPoolExportPointer
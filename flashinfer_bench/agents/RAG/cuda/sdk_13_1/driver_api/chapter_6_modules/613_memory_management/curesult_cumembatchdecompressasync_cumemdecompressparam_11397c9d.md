# CUresult cuMemBatchDecompressAsync (CUmemDecompressParams *paramsArray, size_t count, unsigned int flags, size_t *errorIndex, CUstream stream)

Submit a batch of count independent decompression operations.

###### Parameters

**paramsArray**
The array of structures describing the independent decompression operations.
**count**
The number of entries in paramsArray array.
**flags**
Must be 0.
**errorIndex**
The index into paramsArray of the decompression operation for which the error returned by this
function pertains to. If index is SIZE_MAX and the value returned is not CUDA_SUCCESS, then
the error returned by this function should be considered a general error that does not pertain to a
particular decompression operation. May be NULL, in which case, no index will be recorded in the
event of error.
**stream**
The stream where the work will be enqueued.

###### Returns

CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE,
CUDA_ERROR_INVALID_HANDLE

###### Description

Each of the count decompression operations is described by a single entry in the paramsArray
array. Once the batch has been submitted, the function will return, and decompression will happen
asynchronously w.r.t. the CPU. To the work completion tracking mechanisms in the CUDA driver, the
batch will be considered a single unit of work and processed according to stream semantics, i.e., it is
not possible to query the completion of individual decompression operations within a batch.

The memory pointed to by each of CUmemDecompressParams.src, CUmemDecompressParams.dst,
and CUmemDecompressParams.dstActBytes, must be capable of usage with the


CUDA Driver API TRM-06703-001 _vRelease Version  |  201


Modules


hardware decompress feature. That is, for each of said pointers, the pointer attribute
CU_POINTER_ATTRIBUTE_IS_HW_DECOMPRESS_CAPABLE should give a non-zero
value. To ensure this, the memory backing the pointers should have been allocated using one of
the following CUDA memory allocators: * cuMemAlloc() - cuMemCreate() with the usage flag
CU_MEM_CREATE_USAGE_HW_DECOMPRESS - cuMemAllocFromPoolAsync() from a pool
that was created with the usage flag CU_MEM_POOL_CREATE_USAGE_HW_DECOMPRESS
Additionally, CUmemDecompressParams.src, CUmemDecompressParams.dst, and
CUmemDecompressParams.dstActBytes, must all be accessible from the device associated with the
context where stream was created. For information on how to ensure this, see the documentation for
the allocator of interest.











See also:

cuMemAlloc, cuMemPoolCreate, cuMemAllocFromPoolAsync
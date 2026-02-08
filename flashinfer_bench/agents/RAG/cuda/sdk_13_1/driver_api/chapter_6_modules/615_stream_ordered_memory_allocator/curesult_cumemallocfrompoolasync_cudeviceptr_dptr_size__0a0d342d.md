# CUresult cuMemAllocFromPoolAsync (CUdeviceptr *dptr, size_t bytesize, CUmemoryPool pool, CUstream hStream)

Allocates memory from a specified pool with stream ordered semantics.

###### Parameters

**dptr**

  - Returned device pointer
**bytesize**

  - Number of bytes to allocate
**pool**

  - The pool to allocate from
**hStream**

  - The stream establishing the stream ordering semantic

###### Returns

CUDA_SUCCESS, CUDA_ERROR_INVALID_VALUE, CUDA_ERROR_NOT_INITIALIZED,
CUDA_ERROR_INVALID_CONTEXT (default stream specified with no current context),
CUDA_ERROR_NOT_SUPPORTED, CUDA_ERROR_OUT_OF_MEMORY

###### Description

Inserts an allocation operation into hStream. A pointer to the allocated memory is returned
immediately in *dptr. The allocation must not be accessed until the the allocation operation completes.
The allocation comes from the specified memory pool.


Note:

**‣** The specified memory pool may be from a device different than that of the specified hStream.


Basic stream ordering allows future work submitted into the same stream to use the allocation.

###### **‣**

Stream query, stream synchronize, and CUDA events can be used to guarantee that the allocation
operation completes before work submitted in a separate stream runs.


Note:


During stream capture, this function results in the creation of an allocation node. In this case, the
allocation is owned by the graph instead of the memory pool. The memory pool's properties are used to
set the node's creation parameters.


See also:

cuMemAllocAsync, cuMemFreeAsync, cuDeviceGetDefaultMemPool, cuDeviceGetMemPool,
cuMemPoolCreate, cuMemPoolSetAccess, cuMemPoolSetAttribute


CUDA Driver API TRM-06703-001 _vRelease Version  |  288


Modules
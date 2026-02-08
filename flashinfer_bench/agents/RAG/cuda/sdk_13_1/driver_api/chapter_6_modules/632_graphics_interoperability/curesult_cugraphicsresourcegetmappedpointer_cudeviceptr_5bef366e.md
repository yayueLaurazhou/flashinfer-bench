# CUresult cuGraphicsResourceGetMappedPointer (CUdeviceptr *pDevPtr, size_t *pSize, CUgraphicsResource resource)

Get a device pointer through which to access a mapped graphics resource.

###### Parameters

**pDevPtr**

  - Returned pointer through which resource may be accessed
**pSize**

  - Returned size of the buffer accessible starting at *pPointer
**resource**

  - Mapped resource to access

###### Returns

CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE,
CUDA_ERROR_INVALID_HANDLE, CUDA_ERROR_NOT_MAPPED,
CUDA_ERROR_NOT_MAPPED_AS_POINTER

###### Description

Returns in *pDevPtr a pointer through which the mapped graphics resource resource may be
accessed. Returns in pSize the size of the memory in bytes which may be accessed from that pointer.
The value set in pPointer may change every time that resource is mapped.

If resource is not a buffer then it cannot be accessed via a pointer and
CUDA_ERROR_NOT_MAPPED_AS_POINTER is returned. If resource is not mapped then
CUDA_ERROR_NOT_MAPPED is returned. *


Note:


Note that this function may also return error codes from previous, asynchronous launches.


See also:


CUDA Driver API TRM-06703-001 _vRelease Version  |  558


Modules


cuGraphicsMapResources, cuGraphicsSubResourceGetMappedArray,
cudaGraphicsResourceGetMappedPointer
# CUresult cuGraphicsResourceGetMappedMipmappedArray (CUmipmappedArray *pMipmappedArray, CUgraphicsResource resource)

Get a mipmapped array through which to access a mapped graphics resource.

###### Parameters

**pMipmappedArray**

  - Returned mipmapped array through which resource may be accessed
**resource**

  - Mapped resource to access

###### Returns

CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE,
CUDA_ERROR_INVALID_HANDLE, CUDA_ERROR_NOT_MAPPED,
CUDA_ERROR_NOT_MAPPED_AS_ARRAY

###### Description

Returns in *pMipmappedArray a mipmapped array through which the mapped graphics resource
resource. The value set in *pMipmappedArray may change every time that resource is
mapped.

If resource is not a texture then it cannot be accessed via a mipmapped array and
CUDA_ERROR_NOT_MAPPED_AS_ARRAY is returned. If resource is not mapped then
CUDA_ERROR_NOT_MAPPED is returned.


CUDA Driver API TRM-06703-001 _vRelease Version  |  557


Modules


Note:


Note that this function may also return error codes from previous, asynchronous launches.


See also:

cuGraphicsResourceGetMappedPointer, cudaGraphicsResourceGetMappedMipmappedArray
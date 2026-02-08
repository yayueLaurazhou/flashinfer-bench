# CUresult cuGraphicsSubResourceGetMappedArray (CUarray *pArray, CUgraphicsResource resource, unsigned int arrayIndex, unsigned int mipLevel)

Get an array through which to access a subresource of a mapped graphics resource.

###### Parameters

**pArray**

  - Returned array through which a subresource of resource may be accessed
**resource**

  - Mapped resource to access
**arrayIndex**

  - Array index for array textures or cubemap face index as defined by CUarray_cubemap_face for
cubemap textures for the subresource to access
**mipLevel**

  - Mipmap level for the subresource to access

###### Returns

CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE,
CUDA_ERROR_INVALID_HANDLE, CUDA_ERROR_NOT_MAPPED,
CUDA_ERROR_NOT_MAPPED_AS_ARRAY

###### Description

Returns in *pArray an array through which the subresource of the mapped graphics resource
resource which corresponds to array index arrayIndex and mipmap level mipLevel may be
accessed. The value set in *pArray may change every time that resource is mapped.

If resource is not a texture then it cannot be accessed via an array and
CUDA_ERROR_NOT_MAPPED_AS_ARRAY is returned. If arrayIndex is not a valid array
index for resource then CUDA_ERROR_INVALID_VALUE is returned. If mipLevel is not
a valid mipmap level for resource then CUDA_ERROR_INVALID_VALUE is returned. If
resource is not mapped then CUDA_ERROR_NOT_MAPPED is returned.


Note:


Note that this function may also return error codes from previous, asynchronous launches.


See also:


CUDA Driver API TRM-06703-001 _vRelease Version  |  560


Modules


cuGraphicsResourceGetMappedPointer, cudaGraphicsSubResourceGetMappedArray
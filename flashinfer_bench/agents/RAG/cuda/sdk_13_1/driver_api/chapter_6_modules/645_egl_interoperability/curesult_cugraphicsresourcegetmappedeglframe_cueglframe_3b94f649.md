# CUresult cuGraphicsResourceGetMappedEglFrame (CUeglFrame *eglFrame, CUgraphicsResource resource, unsigned int index, unsigned int mipLevel)

Get an eglFrame through which to access a registered EGL graphics resource.

###### Parameters

**eglFrame**

  - Returned eglFrame.


CUDA Driver API TRM-06703-001 _vRelease Version  |  668


Modules


**resource**

  - Registered resource to access.
**index**

  - Index for cubemap surfaces.
**mipLevel**

  - Mipmap level for the subresource to access.

###### Returns

CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE,
CUDA_ERROR_INVALID_HANDLE, CUDA_ERROR_NOT_MAPPED

###### Description

Returns in *eglFrame an eglFrame pointer through which the registered graphics resource
resource may be accessed. This API can only be called for registered EGL graphics resources.

The CUeglFrame is defined as:


If resource is not registered then CUDA_ERROR_NOT_MAPPED is returned. *


See also:

cuGraphicsMapResources, cuGraphicsSubResourceGetMappedArray,
cuGraphicsResourceGetMappedPointer, cudaGraphicsResourceGetMappedEglFrame


CUDA Driver API TRM-06703-001 _vRelease Version  |  669
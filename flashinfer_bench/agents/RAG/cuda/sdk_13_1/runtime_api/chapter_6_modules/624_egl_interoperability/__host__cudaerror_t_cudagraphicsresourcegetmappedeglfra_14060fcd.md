# __host__cudaError_t cudaGraphicsResourceGetMappedEglFrame (cudaEglFrame *eglFrame, cudaGraphicsResource_t resource, unsigned int index, unsigned int mipLevel)

Get an eglFrame through which to access a registered EGL graphics resource.

##### Parameters

**eglFrame**

  - Returned eglFrame.


CUDA Runtime API vRelease Version  |  294


Modules



**resource**

  - Registered resource to access.
**index**

  - Index for cubemap surfaces.
**mipLevel**

  - Mipmap level for the subresource to access.

##### Returns

cudaSuccess, cudaErrorInvalidValue, cudaErrorUnknown

##### Description

Returns in *eglFrame an eglFrame pointer through which the registered graphics resource
resource may be accessed. This API can only be called for EGL graphics resources.

The cudaEglFrame is defined as





See also:

cudaGraphicsSubResourceGetMappedArray, cudaGraphicsResourceGetMappedPointer,
cuGraphicsResourceGetMappedEglFrame
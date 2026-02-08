# __host__cudaError_t cudaGraphicsResourceGetMappedPointer (void **devPtr, size_t *size, cudaGraphicsResource_t resource)

Get an device pointer through which to access a mapped graphics resource.

##### Parameters

**devPtr**

  - Returned pointer through which resource may be accessed
**size**

  - Returned size of the buffer accessible starting at *devPtr
**resource**

  - Mapped resource to access

##### Returns

cudaSuccess, cudaErrorInvalidValue, cudaErrorInvalidResourceHandle, cudaErrorUnknown

##### Description

Returns in *devPtr a pointer through which the mapped graphics resource resource may be
accessed. Returns in *size the size of the memory in bytes which may be accessed from that pointer.
The value set in devPtr may change every time that resource is mapped.

If resource is not a buffer then it cannot be accessed via a pointer and cudaErrorUnknown is
returned. If resource is not mapped then cudaErrorUnknown is returned. *



See also:

cudaGraphicsMapResources, cudaGraphicsSubResourceGetMappedArray,
cuGraphicsResourceGetMappedPointer


CUDA Runtime API vRelease Version  |  298


Modules
# __host__cudaError_t cudaGLSetBufferObjectMapFlags (GLuint bufObj, unsigned int flags)

Set usage flags for mapping an OpenGL buffer.

##### Parameters

**bufObj**

  - Registered buffer object to set flags for
**flags**

  - Parameters for buffer mapping


CUDA Runtime API vRelease Version  |  238


Modules

##### Returns

cudaSuccess, cudaErrorInvalidValue, cudaErrorInvalidResourceHandle, cudaErrorUnknown

##### Description

Deprecated This function is deprecated as of CUDA 3.0.

Set flags for mapping the OpenGL buffer bufObj

Changes to flags will take effect the next time bufObj is mapped. The flags argument may be any
of the following:

cudaGLMapFlagsNone: Specifies no hints about how this buffer will be used. It is therefore

##### **‣**

assumed that this buffer will be read from and written to by CUDA kernels. This is the default
value.
cudaGLMapFlagsReadOnly: Specifies that CUDA kernels which access this buffer will not write

##### **‣**

to the buffer.
cudaGLMapFlagsWriteDiscard: Specifies that CUDA kernels which access this buffer will not

##### **‣**

read from the buffer and will write over the entire contents of the buffer, so none of the data
previously stored in the buffer will be preserved.

If bufObj has not been registered for use with CUDA, then cudaErrorInvalidResourceHandle is
returned. If bufObj is presently mapped for access by CUDA, then cudaErrorUnknown is returned.


Note:


Note that this function may also return error codes from previous, asynchronous launches.


See also:

cudaGraphicsResourceSetMapFlags
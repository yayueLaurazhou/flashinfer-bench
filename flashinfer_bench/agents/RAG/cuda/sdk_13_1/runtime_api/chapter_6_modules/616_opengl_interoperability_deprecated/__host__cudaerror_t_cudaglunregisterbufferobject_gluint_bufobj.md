# __host__cudaError_t cudaGLUnregisterBufferObject (GLuint bufObj)

Unregisters a buffer object for access by CUDA.

##### Parameters

**bufObj**

  - Buffer object to unregister


CUDA Runtime API vRelease Version  |  241


Modules

##### Returns

cudaSuccess

##### Description

Deprecated This function is deprecated as of CUDA 3.0.

Unregisters the buffer object of ID bufObj for access by CUDA and releases any CUDA resources
associated with the buffer. Once a buffer is unregistered, it may no longer be mapped by CUDA. The
GL context used to create the buffer, or another context from the same share group, must be bound to
the current thread when this is called.


Note:


Note that this function may also return error codes from previous, asynchronous launches.


See also:

cudaGraphicsUnregisterResource
# __host__cudaError_t cudaGLUnmapBufferObjectAsync (GLuint bufObj, cudaStream_t stream)

Unmaps a buffer object for access by CUDA.

##### Parameters

**bufObj**

  - Buffer object to unmap
**stream**

  - Stream to synchronize

##### Returns

cudaSuccess, cudaErrorUnmapBufferObjectFailed

##### Description

Deprecated This function is deprecated as of CUDA 3.0.

Unmaps the buffer object of ID bufObj for access by CUDA. When a buffer is unmapped, the base
address returned by cudaGLMapBufferObject() is invalid and subsequent references to the address
result in undefined behavior. The OpenGL context used to create the buffer, or another context from
the same share group, must be bound to the current thread when this is called.

Stream /p stream is synchronized with the current GL context.


Note:


Note that this function may also return error codes from previous, asynchronous launches.


See also:

cudaGraphicsUnmapResources
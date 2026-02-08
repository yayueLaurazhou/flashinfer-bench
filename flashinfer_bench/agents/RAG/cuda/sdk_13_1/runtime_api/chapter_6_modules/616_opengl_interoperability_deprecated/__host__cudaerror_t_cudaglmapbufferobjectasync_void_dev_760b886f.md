# __host__cudaError_t cudaGLMapBufferObjectAsync (void **devPtr, GLuint bufObj, cudaStream_t stream)

Maps a buffer object for access by CUDA.

##### Parameters

**devPtr**

  - Returned device pointer to CUDA object
**bufObj**

  - Buffer object ID to map
**stream**

  - Stream to synchronize

##### Returns

cudaSuccess, cudaErrorMapBufferObjectFailed

##### Description

Deprecated This function is deprecated as of CUDA 3.0.

Maps the buffer object of ID bufObj into the address space of CUDA and returns in *devPtr the
base pointer of the resulting mapping. The buffer must have previously been registered by calling
cudaGLRegisterBufferObject(). While a buffer is mapped by CUDA, any OpenGL operation which
references the buffer will result in undefined behavior. The OpenGL context used to create the buffer,
or another context from the same share group, must be bound to the current thread when this is called.

Stream /p stream is synchronized with the current GL context.


Note:


Note that this function may also return error codes from previous, asynchronous launches.


CUDA Runtime API vRelease Version  |  237


Modules


See also:

cudaGraphicsMapResources
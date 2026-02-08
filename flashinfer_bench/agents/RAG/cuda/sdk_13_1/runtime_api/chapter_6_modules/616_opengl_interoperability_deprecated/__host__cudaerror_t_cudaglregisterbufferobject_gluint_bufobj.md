# __host__cudaError_t cudaGLRegisterBufferObject (GLuint bufObj)

Registers a buffer object for access by CUDA.

##### Parameters

**bufObj**

  - Buffer object ID to register

##### Returns

cudaSuccess, cudaErrorInitializationError

##### Description

Deprecated This function is deprecated as of CUDA 3.0.

Registers the buffer object of ID bufObj for access by CUDA. This function must be called before
CUDA can map the buffer object. The OpenGL context used to create the buffer, or another context
from the same share group, must be bound to the current thread when this is called.


Note:


Note that this function may also return error codes from previous, asynchronous launches.


See also:

cudaGraphicsGLRegisterBuffer
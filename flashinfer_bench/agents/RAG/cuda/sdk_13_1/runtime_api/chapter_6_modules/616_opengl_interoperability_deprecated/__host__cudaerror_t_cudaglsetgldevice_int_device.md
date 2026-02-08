# __host__cudaError_t cudaGLSetGLDevice (int device)

Sets a CUDA device to use OpenGL interoperability.

##### Parameters

**device**

  - Device to use for OpenGL interoperability

##### Returns

cudaSuccess, cudaErrorInvalidDevice, cudaErrorSetOnActiveProcess

##### Description

Deprecated This function is deprecated as of CUDA 5.0.


CUDA Runtime API vRelease Version  |  239


Modules


This function is deprecated and should no longer be used. It is no longer necessary to associate a
CUDA device with an OpenGL context in order to achieve maximum interoperability performance.

This function will immediately initialize the primary context on device if needed.


Note:


Note that this function may also return error codes from previous, asynchronous launches.


See also:

cudaGraphicsGLRegisterBuffer, cudaGraphicsGLRegisterImage
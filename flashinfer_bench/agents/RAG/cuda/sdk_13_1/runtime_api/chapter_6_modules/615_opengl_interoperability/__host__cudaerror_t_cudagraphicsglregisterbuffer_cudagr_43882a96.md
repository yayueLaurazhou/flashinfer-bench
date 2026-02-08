# __host__cudaError_t cudaGraphicsGLRegisterBuffer (cudaGraphicsResource **resource, GLuint buffer, unsigned int flags)

Registers an OpenGL buffer object.

##### Parameters

**resource**

  - Pointer to the returned object handle
**buffer**

  - name of buffer object to be registered
**flags**

  - Register flags

##### Returns

cudaSuccess, cudaErrorInvalidDevice, cudaErrorInvalidValue, cudaErrorInvalidResourceHandle,
cudaErrorOperatingSystem, cudaErrorUnknown

##### Description

Registers the buffer object specified by buffer for access by CUDA. A handle to the registered
object is returned as resource. The register flags flags specify the intended usage, as follows:

cudaGraphicsRegisterFlagsNone: Specifies no hints about how this resource will be used. It is

##### **‣**

therefore assumed that this resource will be read from and written to by CUDA. This is the default
value.
cudaGraphicsRegisterFlagsReadOnly: Specifies that CUDA will not write to this resource.

##### **‣**

cudaGraphicsRegisterFlagsWriteDiscard: Specifies that CUDA will not read from this resource

##### **‣**

and will write over the entire contents of the resource, so none of the data previously stored in the
resource will be preserved.


Note:


Note that this function may also return error codes from previous, asynchronous launches.


See also:

cudaGraphicsUnregisterResource, cudaGraphicsMapResources,
cudaGraphicsResourceGetMappedPointer, cuGraphicsGLRegisterBuffer


CUDA Runtime API vRelease Version  |  233


Modules
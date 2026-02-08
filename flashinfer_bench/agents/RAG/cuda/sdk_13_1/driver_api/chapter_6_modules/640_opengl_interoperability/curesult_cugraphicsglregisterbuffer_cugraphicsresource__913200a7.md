# CUresult cuGraphicsGLRegisterBuffer (CUgraphicsResource *pCudaResource, GLuint buffer, unsigned int Flags)

Registers an OpenGL buffer object.

###### Parameters

**pCudaResource**

  - Pointer to the returned object handle
**buffer**

  - name of buffer object to be registered
**Flags**

  - Register flags

###### Returns

CUDA_SUCCESS, CUDA_ERROR_INVALID_HANDLE, CUDA_ERROR_ALREADY_MAPPED,
CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_OPERATING_SYSTEM

###### Description

Registers the buffer object specified by buffer for access by CUDA. A handle to the registered
object is returned as pCudaResource. The register flags Flags specify the intended usage, as
follows:

CU_GRAPHICS_REGISTER_FLAGS_NONE: Specifies no hints about how this resource will be

###### **‣**

used. It is therefore assumed that this resource will be read from and written to by CUDA. This is
the default value.
CU_GRAPHICS_REGISTER_FLAGS_READ_ONLY: Specifies that CUDA will not write to this

###### **‣**

resource.
CU_GRAPHICS_REGISTER_FLAGS_WRITE_DISCARD: Specifies that CUDA will not read

###### **‣**

from this resource and will write over the entire contents of the resource, so none of the data
previously stored in the resource will be preserved.


CUDA Driver API TRM-06703-001 _vRelease Version  |  601


Modules


Note:


Note that this function may also return error codes from previous, asynchronous launches.


See also:

cuGraphicsUnregisterResource, cuGraphicsMapResources, cuGraphicsResourceGetMappedPointer,
cudaGraphicsGLRegisterBuffer
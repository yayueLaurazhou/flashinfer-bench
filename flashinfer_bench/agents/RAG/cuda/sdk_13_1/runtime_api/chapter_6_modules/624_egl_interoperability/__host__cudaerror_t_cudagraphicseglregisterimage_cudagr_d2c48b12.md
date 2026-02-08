# __host__cudaError_t cudaGraphicsEGLRegisterImage (cudaGraphicsResource **pCudaResource, EGLImageKHR image, unsigned int flags)

Registers an EGL image.

##### Parameters

**pCudaResource**

  - Pointer to the returned object handle
**image**

  - An EGLImageKHR image which can be used to create target resource.
**flags**

  - Map flags

##### Returns

cudaSuccess, cudaErrorInvalidResourceHandle, cudaErrorInvalidValue, cudaErrorUnknown


CUDA Runtime API vRelease Version  |  293


Modules

##### Description

Registers the EGLImageKHR specified by image for access by CUDA. A handle to the registered
object is returned as pCudaResource. Additional Mapping/Unmapping is not required for the
registered resource and cudaGraphicsResourceGetMappedEglFrame can be directly called on the
pCudaResource.

The application will be responsible for synchronizing access to shared objects. The application must
ensure that any pending operation which access the objects have completed before passing control to
CUDA. This may be accomplished by issuing and waiting for glFinish command on all GLcontexts
(for OpenGL and likewise for other APIs). The application will be also responsible for ensuring that
any pending operation on the registered CUDA resource has completed prior to executing subsequent
commands in other APIs accesing the same memory objects. This can be accomplished by calling
cuCtxSynchronize or cuEventSynchronize (preferably).

The surface's intended usage is specified using flags, as follows:

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

The EGLImageKHR is an object which can be used to create EGLImage target resource. It is defined
as a void pointer. typedef void* EGLImageKHR


See also:

cudaGraphicsUnregisterResource, cudaGraphicsResourceGetMappedEglFrame,
cuGraphicsEGLRegisterImage
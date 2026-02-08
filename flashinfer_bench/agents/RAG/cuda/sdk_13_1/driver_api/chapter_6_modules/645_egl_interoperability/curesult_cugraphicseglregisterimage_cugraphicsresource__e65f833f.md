# CUresult cuGraphicsEGLRegisterImage (CUgraphicsResource *pCudaResource, EGLImageKHR image, unsigned int flags)

Registers an EGL image.

###### Parameters

**pCudaResource**

  - Pointer to the returned object handle
**image**

  - An EGLImageKHR image which can be used to create target resource.
**flags**

  - Map flags

###### Returns

CUDA_SUCCESS, CUDA_ERROR_INVALID_HANDLE, CUDA_ERROR_ALREADY_MAPPED,
CUDA_ERROR_INVALID_CONTEXT,


CUDA Driver API TRM-06703-001 _vRelease Version  |  667


Modules

###### Description

Registers the EGLImageKHR specified by image for access by CUDA. A handle to the registered
object is returned as pCudaResource. Additional Mapping/Unmapping is not required for the
registered resource and cuGraphicsResourceGetMappedEglFrame can be directly called on the
pCudaResource.

The application will be responsible for synchronizing access to shared objects. The application must
ensure that any pending operation which access the objects have completed before passing control to
CUDA. This may be accomplished by issuing and waiting for glFinish command on all GLcontexts
(for OpenGL and likewise for other APIs). The application will be also responsible for ensuring that
any pending operation on the registered CUDA resource has completed prior to executing subsequent
commands in other APIs accesing the same memory objects. This can be accomplished by calling
cuCtxSynchronize or cuEventSynchronize (preferably).

The surface's intended usage is specified using flags, as follows:

CU_GRAPHICS_MAP_RESOURCE_FLAGS_NONE: Specifies no hints about how this resource

###### **‣**

will be used. It is therefore assumed that this resource will be read from and written to by CUDA.
This is the default value.
CU_GRAPHICS_MAP_RESOURCE_FLAGS_READ_ONLY: Specifies that CUDA will not

###### **‣**

write to this resource.
CU_GRAPHICS_MAP_RESOURCE_FLAGS_WRITE_DISCARD: Specifies that CUDA will not

###### **‣**

read from this resource and will write over the entire contents of the resource, so none of the data
previously stored in the resource will be preserved.

The EGLImageKHR is an object which can be used to create EGLImage target resource. It is defined
as a void pointer. typedef void* EGLImageKHR


See also:

cuGraphicsEGLRegisterImage, cuGraphicsUnregisterResource, cuGraphicsResourceSetMapFlags,
cuGraphicsMapResources, cuGraphicsUnmapResources, cudaGraphicsEGLRegisterImage
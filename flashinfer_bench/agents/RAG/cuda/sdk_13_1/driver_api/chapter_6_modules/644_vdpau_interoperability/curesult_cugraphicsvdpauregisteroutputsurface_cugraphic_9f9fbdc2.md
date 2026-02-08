# CUresult cuGraphicsVDPAURegisterOutputSurface (CUgraphicsResource *pCudaResource, VdpOutputSurface vdpSurface, unsigned int flags)

Registers a VDPAU VdpOutputSurface object.

###### Parameters

**pCudaResource**

  - Pointer to the returned object handle


CUDA Driver API TRM-06703-001 _vRelease Version  |  655


Modules


**vdpSurface**

  - The VdpOutputSurface to be registered
**flags**

  - Map flags

###### Returns

CUDA_SUCCESS, CUDA_ERROR_INVALID_HANDLE, CUDA_ERROR_ALREADY_MAPPED,
CUDA_ERROR_INVALID_CONTEXT,

###### Description

Registers the VdpOutputSurface specified by vdpSurface for access by CUDA. A handle to the
registered object is returned as pCudaResource. The surface's intended usage is specified using
flags, as follows:

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

The VdpOutputSurface is presented as an array of subresources that may be accessed using pointers
returned by cuGraphicsSubResourceGetMappedArray. The exact number of valid arrayIndex
values depends on the VDPAU surface format. The mapping is shown in the table below. mipLevel
must be 0.


Note:


Note that this function may also return error codes from previous, asynchronous launches.


See also:

cuCtxCreate, cuVDPAUCtxCreate, cuGraphicsVDPAURegisterVideoSurface,
cuGraphicsUnregisterResource, cuGraphicsResourceSetMapFlags, cuGraphicsMapResources,
cuGraphicsUnmapResources, cuGraphicsSubResourceGetMappedArray, cuVDPAUGetDevice,
cudaGraphicsVDPAURegisterOutputSurface


CUDA Driver API TRM-06703-001 _vRelease Version  |  656


Modules
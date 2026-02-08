# CUresult cuVDPAUCtxCreate (CUcontext *pCtx, unsigned int flags, CUdevice device, VdpDevice vdpDevice, VdpGetProcAddress *vdpGetProcAddress)

Create a CUDA context for interoperability with VDPAU.

###### Parameters

**pCtx**

  - Returned CUDA context
**flags**

  - Options for CUDA context creation
**device**

  - Device on which to create the context
**vdpDevice**

  - The VdpDevice to interop with
**vdpGetProcAddress**

  - VDPAU's VdpGetProcAddress function pointer

###### Returns

CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE,
CUDA_ERROR_OUT_OF_MEMORY

###### Description

Creates a new CUDA context, initializes VDPAU interoperability, and associates the CUDA context
with the calling thread. It must be called before performing any other VDPAU interoperability
operations. It may fail if the needed VDPAU driver facilities are not available. For usage of the flags
parameter, see cuCtxCreate().


Note:


Note that this function may also return error codes from previous, asynchronous launches.


See also:


CUDA Driver API TRM-06703-001 _vRelease Version  |  658


Modules


cuCtxCreate, cuGraphicsVDPAURegisterVideoSurface, cuGraphicsVDPAURegisterOutputSurface,
cuGraphicsUnregisterResource, cuGraphicsResourceSetMapFlags, cuGraphicsMapResources,
cuGraphicsUnmapResources, cuGraphicsSubResourceGetMappedArray, cuVDPAUGetDevice
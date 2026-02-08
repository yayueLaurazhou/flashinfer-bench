# CUresult cuVDPAUGetDevice (CUdevice *pDevice, VdpDevice vdpDevice, VdpGetProcAddress *vdpGetProcAddress)

Gets the CUDA device associated with a VDPAU device.

###### Parameters

**pDevice**

  - Device associated with vdpDevice
**vdpDevice**

  - A VdpDevice handle
**vdpGetProcAddress**

  - VDPAU's VdpGetProcAddress function pointer

###### Returns

CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE

###### Description

Returns in *pDevice the CUDA device associated with a vdpDevice, if applicable.


Note:


Note that this function may also return error codes from previous, asynchronous launches.


See also:

cuCtxCreate, cuVDPAUCtxCreate, cuGraphicsVDPAURegisterVideoSurface,
cuGraphicsVDPAURegisterOutputSurface, cuGraphicsUnregisterResource,
cuGraphicsResourceSetMapFlags, cuGraphicsMapResources, cuGraphicsUnmapResources,
cuGraphicsSubResourceGetMappedArray, cudaVDPAUGetDevice
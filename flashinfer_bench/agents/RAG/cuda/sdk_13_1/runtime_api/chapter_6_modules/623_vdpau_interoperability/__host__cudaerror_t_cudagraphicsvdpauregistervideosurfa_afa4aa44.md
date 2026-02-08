# __host__cudaError_t cudaGraphicsVDPAURegisterVideoSurface (cudaGraphicsResource **resource, VdpVideoSurface vdpSurface, unsigned int flags)

Register a VdpVideoSurface object.

##### Parameters

**resource**

  - Pointer to the returned object handle
**vdpSurface**

  - VDPAU object to be registered


CUDA Runtime API vRelease Version  |  283


Modules


**flags**

  - Map flags

##### Returns

cudaSuccess, cudaErrorInvalidDevice, cudaErrorInvalidValue, cudaErrorInvalidResourceHandle,
cudaErrorUnknown

##### Description

Registers the VdpVideoSurface specified by vdpSurface for access by CUDA. A handle to the
registered object is returned as resource. The surface's intended usage is specified using flags, as
follows:

cudaGraphicsMapFlagsNone: Specifies no hints about how this resource will be used. It is

##### **‣**

therefore assumed that this resource will be read from and written to by CUDA. This is the default
value.
cudaGraphicsMapFlagsReadOnly: Specifies that CUDA will not write to this resource.

##### **‣**

cudaGraphicsMapFlagsWriteDiscard: Specifies that CUDA will not read from this resource and

##### **‣**

will write over the entire contents of the resource, so none of the data previously stored in the
resource will be preserved.


Note:


Note that this function may also return error codes from previous, asynchronous launches.


See also:

cudaVDPAUSetVDPAUDevice, cudaGraphicsUnregisterResource,
cudaGraphicsSubResourceGetMappedArray, cuGraphicsVDPAURegisterVideoSurface
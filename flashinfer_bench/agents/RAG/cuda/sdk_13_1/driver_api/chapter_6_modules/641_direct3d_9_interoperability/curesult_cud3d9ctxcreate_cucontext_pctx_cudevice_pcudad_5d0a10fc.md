# CUresult cuD3D9CtxCreate (CUcontext *pCtx, CUdevice *pCudaDevice, unsigned int Flags, IDirect3DDevice9 *pD3DDevice)

Create a CUDA context for interoperability with Direct3D 9.

###### Parameters

**pCtx**

  - Returned newly created CUDA context
**pCudaDevice**

  - Returned pointer to the device on which the context was created
**Flags**

  - Context creation flags (see cuCtxCreate() for details)
**pD3DDevice**

  - Direct3D device to create interoperability context with

###### Returns

CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
CUDA_ERROR_INVALID_VALUE, CUDA_ERROR_OUT_OF_MEMORY,
CUDA_ERROR_UNKNOWN

###### Description

Creates a new CUDA context, enables interoperability for that context with the Direct3D device
pD3DDevice, and associates the created CUDA context with the calling thread. The created
CUcontext will be returned in *pCtx. Direct3D resources from this device may be registered
and mapped through the lifetime of this CUDA context. If pCudaDevice is non-NULL then the
CUdevice on which this CUDA context was created will be returned in *pCudaDevice.


CUDA Driver API TRM-06703-001 _vRelease Version  |  612


Modules


On success, this call will increase the internal reference count on pD3DDevice. This reference count
will be decremented upon destruction of this context through cuCtxDestroy(). This context will cease
to function if pD3DDevice is destroyed or encounters an error.

Note that this function is never required for correct functionality. Use of this function will result in
accelerated interoperability only when the operating system is Windows Vista or Windows 7, and the
device pD3DDdevice is not an IDirect3DDevice9Ex. In all other circumstances, this function is not
necessary.


Note:


Note that this function may also return error codes from previous, asynchronous launches.


See also:

cuD3D9GetDevice, cuGraphicsD3D9RegisterResource
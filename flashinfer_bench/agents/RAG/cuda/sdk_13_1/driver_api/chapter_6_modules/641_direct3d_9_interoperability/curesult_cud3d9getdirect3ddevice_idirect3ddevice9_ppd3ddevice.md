# CUresult cuD3D9GetDirect3DDevice (IDirect3DDevice9 **ppD3DDevice)

Get the Direct3D 9 device against which the current CUDA context was created.

###### Parameters

**ppD3DDevice**

  - Returned Direct3D device corresponding to CUDA context

###### Returns

CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
CUDA_ERROR_INVALID_CONTEXT CUDA_ERROR_INVALID_GRAPHICS_CONTEXT

###### Description

Returns in *ppD3DDevice the Direct3D device against which this CUDA context was created in
cuD3D9CtxCreate().


Note:


Note that this function may also return error codes from previous, asynchronous launches.


See also:

cuD3D9GetDevice, cudaD3D9GetDirect3DDevice


CUDA Driver API TRM-06703-001 _vRelease Version  |  616


Modules
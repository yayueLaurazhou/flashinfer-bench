# __host__cudaError_t cudaD3D9GetDirect3DDevice (IDirect3DDevice9 **ppD3D9Device)

Gets the Direct3D device against which the current CUDA context was created.

##### Parameters

**ppD3D9Device**

  - Returns the Direct3D device for this thread

##### Returns

cudaSuccess, cudaErrorInvalidGraphicsContext, cudaErrorUnknown


CUDA Runtime API vRelease Version  |  244


Modules

##### Description

Returns in *ppD3D9Device the Direct3D device against which this CUDA context was created in
cudaD3D9SetDirect3DDevice().


Note:


Note that this function may also return error codes from previous, asynchronous launches.


See also:

cudaD3D9SetDirect3DDevice, cuD3D9GetDirect3DDevice
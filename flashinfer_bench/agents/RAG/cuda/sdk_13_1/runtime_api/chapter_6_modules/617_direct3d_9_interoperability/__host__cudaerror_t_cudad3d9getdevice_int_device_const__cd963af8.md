# __host__cudaError_t cudaD3D9GetDevice (int *device, const char *pszAdapterName)

Gets the device number for an adapter.

##### Parameters

**device**

  - Returns the device corresponding to pszAdapterName
**pszAdapterName**

  - D3D9 adapter to get device for

##### Returns

cudaSuccess, cudaErrorInvalidValue, cudaErrorUnknown

##### Description

Returns in *device the CUDA-compatible device corresponding to the adapter name
pszAdapterName obtained from EnumDisplayDevices or IDirect3D9::GetAdapterIdentifier(). If no
device on the adapter with name pszAdapterName is CUDA-compatible then the call will fail.


Note:


Note that this function may also return error codes from previous, asynchronous launches.


See also:

cudaD3D9SetDirect3DDevice, cudaGraphicsD3D9RegisterResource, cuD3D9GetDevice
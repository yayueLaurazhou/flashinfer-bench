# CUresult cuD3D9GetDevice (CUdevice *pCudaDevice, const char *pszAdapterName)

Gets the CUDA device corresponding to a display adapter.

###### Parameters

**pCudaDevice**

  - Returned CUDA device corresponding to pszAdapterName
**pszAdapterName**

  - Adapter name to query for device

###### Returns

CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
CUDA_ERROR_INVALID_VALUE, CUDA_ERROR_NOT_FOUND,
CUDA_ERROR_UNKNOWN

###### Description

Returns in *pCudaDevice the CUDA-compatible device corresponding to the adapter name
pszAdapterName obtained from EnumDisplayDevices() or IDirect3D9::GetAdapterIdentifier().

If no device on the adapter with name pszAdapterName is CUDA-compatible, then the call will
fail.


CUDA Driver API TRM-06703-001 _vRelease Version  |  614


Modules


Note:


Note that this function may also return error codes from previous, asynchronous launches.


See also:

cuD3D9CtxCreate, cudaD3D9GetDevice
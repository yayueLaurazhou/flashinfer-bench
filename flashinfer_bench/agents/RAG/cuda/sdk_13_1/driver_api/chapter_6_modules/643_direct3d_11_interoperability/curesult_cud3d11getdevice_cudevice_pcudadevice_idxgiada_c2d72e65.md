# CUresult cuD3D11GetDevice (CUdevice *pCudaDevice, IDXGIAdapter *pAdapter)

Gets the CUDA device corresponding to a display adapter.

###### Parameters

**pCudaDevice**

  - Returned CUDA device corresponding to pAdapter
**pAdapter**

  - Adapter to query for CUDA device

###### Returns

CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
CUDA_ERROR_NO_DEVICE, CUDA_ERROR_INVALID_VALUE,
CUDA_ERROR_NOT_FOUND, CUDA_ERROR_UNKNOWN

###### Description

Returns in *pCudaDevice the CUDA-compatible device corresponding to the adapter pAdapter
obtained from IDXGIFactory::EnumAdapters.

If no device on pAdapter is CUDA-compatible the call will return CUDA_ERROR_NO_DEVICE.


Note:


Note that this function may also return error codes from previous, asynchronous launches.


See also:

cuD3D11GetDevices, cudaD3D11GetDevice
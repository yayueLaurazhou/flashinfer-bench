# CUresult cuD3D9GetDevices (unsigned int *pCudaDeviceCount, CUdevice *pCudaDevices, unsigned int cudaDeviceCount, IDirect3DDevice9 *pD3D9Device, CUd3d9DeviceList deviceList)

Gets the CUDA devices corresponding to a Direct3D 9 device.

###### Parameters

**pCudaDeviceCount**

  - Returned number of CUDA devices corresponding to pD3D9Device
**pCudaDevices**

  - Returned CUDA devices corresponding to pD3D9Device
**cudaDeviceCount**

  - The size of the output device array pCudaDevices
**pD3D9Device**

  - Direct3D 9 device to query for CUDA devices
**deviceList**

  - The set of devices to return. This set may be CU_D3D9_DEVICE_LIST_ALL for all devices,
CU_D3D9_DEVICE_LIST_CURRENT_FRAME for the devices used to render the current frame
(in SLI), or CU_D3D9_DEVICE_LIST_NEXT_FRAME for the devices used to render the next
frame (in SLI).

###### Returns

CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
CUDA_ERROR_NO_DEVICE, CUDA_ERROR_INVALID_VALUE,
CUDA_ERROR_NOT_FOUND, CUDA_ERROR_UNKNOWN

###### Description

Returns in *pCudaDeviceCount the number of CUDA-compatible device corresponding to the
Direct3D 9 device pD3D9Device. Also returns in *pCudaDevices at most cudaDeviceCount
of the CUDA-compatible devices corresponding to the Direct3D 9 device pD3D9Device.

If any of the GPUs being used to render pDevice are not CUDA capable then the call will return
CUDA_ERROR_NO_DEVICE.


CUDA Driver API TRM-06703-001 _vRelease Version  |  615


Modules


Note:


Note that this function may also return error codes from previous, asynchronous launches.


See also:

cuD3D9CtxCreate, cudaD3D9GetDevices
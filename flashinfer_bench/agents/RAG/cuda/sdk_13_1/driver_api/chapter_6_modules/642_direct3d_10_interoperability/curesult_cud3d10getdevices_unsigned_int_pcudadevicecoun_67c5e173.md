# CUresult cuD3D10GetDevices (unsigned int *pCudaDeviceCount, CUdevice *pCudaDevices, unsigned int cudaDeviceCount, ID3D10Device *pD3D10Device, CUd3d10DeviceList deviceList)

Gets the CUDA devices corresponding to a Direct3D 10 device.

###### Parameters

**pCudaDeviceCount**

  - Returned number of CUDA devices corresponding to pD3D10Device
**pCudaDevices**

  - Returned CUDA devices corresponding to pD3D10Device
**cudaDeviceCount**

  - The size of the output device array pCudaDevices
**pD3D10Device**

  - Direct3D 10 device to query for CUDA devices
**deviceList**

  - The set of devices to return. This set may be CU_D3D10_DEVICE_LIST_ALL for all devices,
CU_D3D10_DEVICE_LIST_CURRENT_FRAME for the devices used to render the current frame
(in SLI), or CU_D3D10_DEVICE_LIST_NEXT_FRAME for the devices used to render the next
frame (in SLI).

###### Returns

CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
CUDA_ERROR_NO_DEVICE, CUDA_ERROR_INVALID_VALUE,
CUDA_ERROR_NOT_FOUND, CUDA_ERROR_UNKNOWN

###### Description

Returns in *pCudaDeviceCount the number of CUDA-compatible device corresponding
to the Direct3D 10 device pD3D10Device. Also returns in *pCudaDevices at most
cudaDeviceCount of the CUDA-compatible devices corresponding to the Direct3D 10 device
pD3D10Device.

If any of the GPUs being used to render pDevice are not CUDA capable then the call will return
CUDA_ERROR_NO_DEVICE.


Note:


Note that this function may also return error codes from previous, asynchronous launches.


See also:


CUDA Driver API TRM-06703-001 _vRelease Version  |  632


Modules


cuD3D10GetDevice, cudaD3D10GetDevices
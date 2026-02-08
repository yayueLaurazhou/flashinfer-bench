# __host__cudaError_t cudaD3D9GetDevices (unsigned int *pCudaDeviceCount, int *pCudaDevices, unsigned int cudaDeviceCount, IDirect3DDevice9 *pD3D9Device, cudaD3D9DeviceList deviceList)

Gets the CUDA devices corresponding to a Direct3D 9 device.

##### Parameters

**pCudaDeviceCount**

  - Returned number of CUDA devices corresponding to pD3D9Device
**pCudaDevices**

  - Returned CUDA devices corresponding to pD3D9Device


CUDA Runtime API vRelease Version  |  243


Modules


**cudaDeviceCount**

  - The size of the output device array pCudaDevices
**pD3D9Device**

  - Direct3D 9 device to query for CUDA devices
**deviceList**

  - The set of devices to return. This set may be cudaD3D9DeviceListAll for all devices,
cudaD3D9DeviceListCurrentFrame for the devices used to render the current frame (in SLI), or
cudaD3D9DeviceListNextFrame for the devices used to render the next frame (in SLI).

##### Returns

cudaSuccess, cudaErrorNoDevice, cudaErrorUnknown

##### Description

Returns in *pCudaDeviceCount the number of CUDA-compatible devices corresponding to the
Direct3D 9 device pD3D9Device. Also returns in *pCudaDevices at most cudaDeviceCount
of the the CUDA-compatible devices corresponding to the Direct3D 9 device pD3D9Device.

If any of the GPUs being used to render pDevice are not CUDA capable then the call will return
cudaErrorNoDevice.


Note:


Note that this function may also return error codes from previous, asynchronous launches.


See also:

cudaGraphicsUnregisterResource, cudaGraphicsMapResources,
cudaGraphicsSubResourceGetMappedArray, cudaGraphicsResourceGetMappedPointer,
cuD3D9GetDevices
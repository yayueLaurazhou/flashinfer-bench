# __host__cudaError_t cudaD3D10GetDevices (unsigned int *pCudaDeviceCount, int *pCudaDevices, unsigned int cudaDeviceCount, ID3D10Device *pD3D10Device, cudaD3D10DeviceList deviceList)

Gets the CUDA devices corresponding to a Direct3D 10 device.

##### Parameters

**pCudaDeviceCount**

  - Returned number of CUDA devices corresponding to pD3D10Device
**pCudaDevices**

  - Returned CUDA devices corresponding to pD3D10Device
**cudaDeviceCount**

  - The size of the output device array pCudaDevices
**pD3D10Device**

  - Direct3D 10 device to query for CUDA devices
**deviceList**

  - The set of devices to return. This set may be cudaD3D10DeviceListAll for all devices,
cudaD3D10DeviceListCurrentFrame for the devices used to render the current frame (in SLI), or
cudaD3D10DeviceListNextFrame for the devices used to render the next frame (in SLI).

##### Returns

cudaSuccess, cudaErrorNoDevice, cudaErrorUnknown

##### Description

Returns in *pCudaDeviceCount the number of CUDA-compatible devices corresponding
to the Direct3D 10 device pD3D10Device. Also returns in *pCudaDevices at most
cudaDeviceCount of the the CUDA-compatible devices corresponding to the Direct3D 10 device
pD3D10Device.

If any of the GPUs being used to render pDevice are not CUDA capable then the call will return
cudaErrorNoDevice.


Note:


Note that this function may also return error codes from previous, asynchronous launches.


See also:


CUDA Runtime API vRelease Version  |  261


Modules


cudaGraphicsUnregisterResource, cudaGraphicsMapResources,
cudaGraphicsSubResourceGetMappedArray, cudaGraphicsResourceGetMappedPointer,
cuD3D10GetDevices
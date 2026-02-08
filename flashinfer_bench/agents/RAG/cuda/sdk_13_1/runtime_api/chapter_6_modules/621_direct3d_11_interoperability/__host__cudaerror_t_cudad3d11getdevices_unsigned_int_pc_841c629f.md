# __host__cudaError_t cudaD3D11GetDevices (unsigned int *pCudaDeviceCount, int *pCudaDevices, unsigned int cudaDeviceCount, ID3D11Device *pD3D11Device, cudaD3D11DeviceList deviceList)

Gets the CUDA devices corresponding to a Direct3D 11 device.

##### Parameters

**pCudaDeviceCount**

  - Returned number of CUDA devices corresponding to pD3D11Device


CUDA Runtime API vRelease Version  |  277


Modules


**pCudaDevices**

  - Returned CUDA devices corresponding to pD3D11Device
**cudaDeviceCount**

  - The size of the output device array pCudaDevices
**pD3D11Device**

  - Direct3D 11 device to query for CUDA devices
**deviceList**

  - The set of devices to return. This set may be cudaD3D11DeviceListAll for all devices,
cudaD3D11DeviceListCurrentFrame for the devices used to render the current frame (in SLI), or
cudaD3D11DeviceListNextFrame for the devices used to render the next frame (in SLI).

##### Returns

cudaSuccess, cudaErrorNoDevice, cudaErrorUnknown

##### Description

Returns in *pCudaDeviceCount the number of CUDA-compatible devices corresponding
to the Direct3D 11 device pD3D11Device. Also returns in *pCudaDevices at most
cudaDeviceCount of the the CUDA-compatible devices corresponding to the Direct3D 11 device
pD3D11Device.

If any of the GPUs being used to render pDevice are not CUDA capable then the call will return
cudaErrorNoDevice.


Note:


Note that this function may also return error codes from previous, asynchronous launches.


See also:

cudaGraphicsUnregisterResource, cudaGraphicsMapResources,
cudaGraphicsSubResourceGetMappedArray, cudaGraphicsResourceGetMappedPointer,
cuD3D11GetDevices


CUDA Runtime API vRelease Version  |  278


Modules
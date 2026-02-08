# __host__cudaError_t cudaD3D11SetDirect3DDevice (ID3D11Device *pD3D11Device, int device)

Sets the Direct3D 11 device to use for interoperability with a CUDA device.

##### Parameters

**pD3D11Device**

  - Direct3D device to use for interoperability


CUDA Runtime API vRelease Version  |  281


Modules


**device**

  - The CUDA device to use. This device must be among the devices returned when querying
cudaD3D11DeviceListAll from cudaD3D11GetDevices, may be set to -1 to automatically select an
appropriate CUDA device.

##### Returns

cudaSuccess, cudaErrorInitializationError, cudaErrorInvalidValue, cudaErrorSetOnActiveProcess

##### Description

Deprecated This function is deprecated as of CUDA 5.0.

This function is deprecated and should no longer be used. It is no longer necessary to associate a
CUDA device with a D3D11 device in order to achieve maximum interoperability performance.

This function will immediately initialize the primary context on device if needed.


Note:


Note that this function may also return error codes from previous, asynchronous launches.


See also:

cudaD3D11GetDevice, cudaGraphicsD3D11RegisterResource, cudaDeviceReset
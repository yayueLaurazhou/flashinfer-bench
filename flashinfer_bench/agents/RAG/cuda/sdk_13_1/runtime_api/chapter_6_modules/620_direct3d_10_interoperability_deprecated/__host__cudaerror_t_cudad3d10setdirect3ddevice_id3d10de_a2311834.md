# __host__cudaError_t cudaD3D10SetDirect3DDevice (ID3D10Device *pD3D10Device, int device)

Sets the Direct3D 10 device to use for interoperability with a CUDA device.

##### Parameters

**pD3D10Device**

  - Direct3D device to use for interoperability
**device**

  - The CUDA device to use. This device must be among the devices returned when querying
cudaD3D10DeviceListAll from cudaD3D10GetDevices, may be set to -1 to automatically select an
appropriate CUDA device.

##### Returns

cudaSuccess, cudaErrorInitializationError, cudaErrorInvalidValue, cudaErrorSetOnActiveProcess

##### Description

Deprecated This function is deprecated as of CUDA 5.0.

This function is deprecated and should no longer be used. It is no longer necessary to associate a
CUDA device with a D3D10 device in order to achieve maximum interoperability performance.

This function will immediately initialize the primary context on device if needed.


Note:


Note that this function may also return error codes from previous, asynchronous launches.


See also:

cudaD3D10GetDevice, cudaGraphicsD3D10RegisterResource, cudaDeviceReset


CUDA Runtime API vRelease Version  |  274


Modules
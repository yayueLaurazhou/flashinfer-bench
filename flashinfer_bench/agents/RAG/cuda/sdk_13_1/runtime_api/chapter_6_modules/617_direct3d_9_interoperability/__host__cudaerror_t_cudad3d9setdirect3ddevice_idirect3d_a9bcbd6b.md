# __host__cudaError_t cudaD3D9SetDirect3DDevice (IDirect3DDevice9 *pD3D9Device, int device)

Sets the Direct3D 9 device to use for interoperability with a CUDA device.

##### Parameters

**pD3D9Device**

  - Direct3D device to use for this thread
**device**

  - The CUDA device to use. This device must be among the devices returned when querying
cudaD3D9DeviceListAll from cudaD3D9GetDevices, may be set to -1 to automatically select an
appropriate CUDA device.

##### Returns

cudaSuccess, cudaErrorInitializationError, cudaErrorInvalidValue, cudaErrorSetOnActiveProcess

##### Description

Records pD3D9Device as the Direct3D 9 device to use for Direct3D 9 interoperability with the
CUDA device device and sets device as the current device for the calling host thread.

This function will immediately initialize the primary context on device if needed.

If device has already been initialized then this call will fail with the error
cudaErrorSetOnActiveProcess. In this case it is necessary to reset device using cudaDeviceReset()
before Direct3D 9 interoperability on device may be enabled.

Successfully initializing CUDA interoperability with pD3D9Device will increase the internal
reference count on pD3D9Device. This reference count will be decremented when device is reset
using cudaDeviceReset().


CUDA Runtime API vRelease Version  |  245


Modules


Note that this function is never required for correct functionality. Use of this function will result in
accelerated interoperability only when the operating system is Windows Vista or Windows 7, and the
device pD3DDdevice is not an IDirect3DDevice9Ex. In all other cirumstances, this function is not
necessary.


Note:


Note that this function may also return error codes from previous, asynchronous launches.


See also:

cudaD3D9GetDevice, cudaGraphicsD3D9RegisterResource, cudaDeviceReset
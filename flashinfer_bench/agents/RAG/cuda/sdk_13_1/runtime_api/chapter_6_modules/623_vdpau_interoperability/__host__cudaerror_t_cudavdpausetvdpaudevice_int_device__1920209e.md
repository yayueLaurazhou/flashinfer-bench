# __host__cudaError_t cudaVDPAUSetVDPAUDevice (int device, VdpDevice vdpDevice, VdpGetProcAddress *vdpGetProcAddress)

Sets a CUDA device to use VDPAU interoperability.

##### Parameters

**device**

  - Device to use for VDPAU interoperability
**vdpDevice**

  - The VdpDevice to interoperate with
**vdpGetProcAddress**

  - VDPAU's VdpGetProcAddress function pointer

##### Returns

cudaSuccess, cudaErrorInvalidDevice, cudaErrorSetOnActiveProcess

##### Description

Records vdpDevice as the VdpDevice for VDPAU interoperability with the CUDA device device
and sets device as the current device for the calling host thread.

This function will immediately initialize the primary context on device if needed.


CUDA Runtime API vRelease Version  |  285


Modules


If device has already been initialized then this call will fail with the error
cudaErrorSetOnActiveProcess. In this case it is necessary to reset device using cudaDeviceReset()
before VDPAU interoperability on device may be enabled.


Note:


Note that this function may also return error codes from previous, asynchronous launches.


See also:

cudaGraphicsVDPAURegisterVideoSurface, cudaGraphicsVDPAURegisterOutputSurface,
cudaDeviceReset
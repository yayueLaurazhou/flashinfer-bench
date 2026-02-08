# __host__cudaError_t cudaVDPAUGetDevice (int *device, VdpDevice vdpDevice, VdpGetProcAddress *vdpGetProcAddress)

Gets the CUDA device associated with a VdpDevice.

##### Parameters

**device**

  - Returns the device associated with vdpDevice, or -1 if the device associated with vdpDevice is not
a compute device.
**vdpDevice**

  - A VdpDevice handle


CUDA Runtime API vRelease Version  |  284


Modules


**vdpGetProcAddress**

  - VDPAU's VdpGetProcAddress function pointer

##### Returns

cudaSuccess

##### Description

Returns the CUDA device associated with a VdpDevice, if applicable.


Note:


Note that this function may also return error codes from previous, asynchronous launches.


See also:

cudaVDPAUSetVDPAUDevice, cuVDPAUGetDevice
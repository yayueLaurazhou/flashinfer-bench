# __host__cudaError_t cudaDeviceEnablePeerAccess (int peerDevice, unsigned int flags)

Enables direct access to memory allocations on a peer device.

##### Parameters

**peerDevice**

  - Peer device to enable direct access to from the current device
**flags**

  - Reserved for future use and must be set to 0

##### Returns

cudaSuccess, cudaErrorInvalidDevice, cudaErrorPeerAccessAlreadyEnabled, cudaErrorInvalidValue

##### Description

On success, all allocations from peerDevice will immediately be accessible by the current device.
They will remain accessible until access is explicitly disabled using cudaDeviceDisablePeerAccess() or
either device is reset using cudaDeviceReset().

Note that access granted by this call is unidirectional and that in order to access memory on the current
device from peerDevice, a separate symmetric call to cudaDeviceEnablePeerAccess() is required.


CUDA Runtime API vRelease Version  |  230


Modules


Note that there are both device-wide and system-wide limitations per system configuration, as noted in
the CUDA Programming Guide under the section "Peer-to-Peer Memory Access".

Returns cudaErrorInvalidDevice if cudaDeviceCanAccessPeer() indicates that the current device
cannot directly access memory from peerDevice.

Returns cudaErrorPeerAccessAlreadyEnabled if direct access of peerDevice from the current
device has already been enabled.

Returns cudaErrorInvalidValue if flags is not 0.



See also:

cudaDeviceCanAccessPeer, cudaDeviceDisablePeerAccess, cuCtxEnablePeerAccess
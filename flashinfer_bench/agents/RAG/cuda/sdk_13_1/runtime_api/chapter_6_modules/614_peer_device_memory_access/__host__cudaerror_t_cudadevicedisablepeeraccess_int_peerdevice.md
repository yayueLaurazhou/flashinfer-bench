# __host__cudaError_t cudaDeviceDisablePeerAccess (int peerDevice)

Disables direct access to memory allocations on a peer device.

##### Parameters

**peerDevice**

  - Peer device to disable direct access to


CUDA Runtime API vRelease Version  |  229


Modules

##### Returns

cudaSuccess, cudaErrorPeerAccessNotEnabled, cudaErrorInvalidDevice

##### Description

Returns cudaErrorPeerAccessNotEnabled if direct access to memory on peerDevice has not yet
been enabled from the current device.



See also:

cudaDeviceCanAccessPeer, cudaDeviceEnablePeerAccess, cuCtxDisablePeerAccess
# __host__cudaError_t cudaDeviceCanAccessPeer (int *canAccessPeer, int device, int peerDevice)

Queries if a device may directly access a peer device's memory.

##### Parameters

**canAccessPeer**

  - Returned access capability
**device**

  - Device from which allocations on peerDevice are to be directly accessed.
**peerDevice**

  - Device on which the allocations to be directly accessed by device reside.

##### Returns

cudaSuccess, cudaErrorInvalidDevice

##### Description

Returns in *canAccessPeer a value of 1 if device device is capable of directly accessing
memory from peerDevice and 0 otherwise. If direct access of peerDevice from device is
possible, then access may be enabled by calling cudaDeviceEnablePeerAccess().



See also:

cudaDeviceEnablePeerAccess, cudaDeviceDisablePeerAccess, cuDeviceCanAccessPeer
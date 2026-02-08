# CUresult cuDeviceCanAccessPeer (int *canAccessPeer, CUdevice dev, CUdevice peerDev)

Queries if a device may directly access a peer device's memory.

###### Parameters

**canAccessPeer**

  - Returned access capability
**dev**

  - Device from which allocations on peerDev are to be directly accessed.
**peerDev**

  - Device on which the allocations to be directly accessed by dev reside.

###### Returns

CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
CUDA_ERROR_INVALID_DEVICE

###### Description

Returns in *canAccessPeer a value of 1 if contexts on dev are capable of directly accessing
memory from contexts on peerDev and 0 otherwise. If direct access of peerDev from dev is
possible, then access may be enabled on two specific contexts by calling cuCtxEnablePeerAccess().


Note:


Note that this function may also return error codes from previous, asynchronous launches.


See also:

cuCtxEnablePeerAccess, cuCtxDisablePeerAccess, cudaDeviceCanAccessPeer


CUDA Driver API TRM-06703-001 _vRelease Version  |  553


Modules
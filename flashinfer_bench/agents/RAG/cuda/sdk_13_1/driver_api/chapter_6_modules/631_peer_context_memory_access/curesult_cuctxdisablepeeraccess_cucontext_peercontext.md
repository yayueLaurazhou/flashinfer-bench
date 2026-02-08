# CUresult cuCtxDisablePeerAccess (CUcontext peerContext)

Disables direct access to memory allocations in a peer context and unregisters any registered
allocations.

###### Parameters

**peerContext**

  - Peer context to disable direct access to

###### Returns

CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
CUDA_ERROR_PEER_ACCESS_NOT_ENABLED, CUDA_ERROR_INVALID_CONTEXT,

###### Description

Returns CUDA_ERROR_PEER_ACCESS_NOT_ENABLED if direct peer access has not yet been
enabled from peerContext to the current context.

Returns CUDA_ERROR_INVALID_CONTEXT if there is no current context, or if peerContext is
not a valid context.


Note:


Note that this function may also return error codes from previous, asynchronous launches.


CUDA Driver API TRM-06703-001 _vRelease Version  |  551


Modules


See also:

cuDeviceCanAccessPeer, cuCtxEnablePeerAccess, cudaDeviceDisablePeerAccess
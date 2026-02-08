# CUresult cuCtxEnablePeerAccess (CUcontext peerContext, unsigned int Flags)

Enables direct access to memory allocations in a peer context.

###### Parameters

**peerContext**

  - Peer context to enable direct access to from the current context
**Flags**

  - Reserved for future use and must be set to 0

###### Returns

CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED, CUDA_ERROR_TOO_MANY_PEERS,
CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_PEER_ACCESS_UNSUPPORTED,
CUDA_ERROR_INVALID_VALUE

###### Description

If both the current context and peerContext are on devices which support unified addressing (as
may be queried using CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING) and same major
compute capability, then on success all allocations from peerContext will immediately be
accessible by the current context. See Unified Addressing for additional details.

Note that access granted by this call is unidirectional and that in order to access memory from the
current context in peerContext, a separate symmetric call to cuCtxEnablePeerAccess() is required.

Note that there are both device-wide and system-wide limitations per system configuration, as noted in
the CUDA Programming Guide under the section "Peer-to-Peer Memory Access".

Returns CUDA_ERROR_PEER_ACCESS_UNSUPPORTED if cuDeviceCanAccessPeer() indicates
that the CUdevice of the current context cannot directly access memory from the CUdevice of
peerContext.

Returns CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED if direct access of
peerContext from the current context has already been enabled.

Returns CUDA_ERROR_TOO_MANY_PEERS if direct peer access is not possible because hardware
resources required for peer access have been exhausted.

Returns CUDA_ERROR_INVALID_CONTEXT if there is no current context, peerContext is not
a valid context, or if the current context is peerContext.

Returns CUDA_ERROR_INVALID_VALUE if Flags is not 0.


CUDA Driver API TRM-06703-001 _vRelease Version  |  552


Modules


Note:


Note that this function may also return error codes from previous, asynchronous launches.


See also:

cuDeviceCanAccessPeer, cuCtxDisablePeerAccess, cudaDeviceEnablePeerAccess
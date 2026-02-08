# CUresult cuIpcOpenMemHandle (CUdeviceptr *pdptr, CUipcMemHandle handle, unsigned int Flags)

Opens an interprocess memory handle exported from another process and returns a device pointer
usable in the local process.

###### Parameters

**pdptr**

  - Returned device pointer
**handle**

  - CUipcMemHandle to open
**Flags**

  - Flags for this operation. Must be specified as CU_IPC_MEM_LAZY_ENABLE_PEER_ACCESS

###### Returns

CUDA_SUCCESS, CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_MAP_FAILED,
CUDA_ERROR_INVALID_HANDLE, CUDA_ERROR_TOO_MANY_PEERS,
CUDA_ERROR_INVALID_VALUE

###### Description

Maps memory exported from another process with cuIpcGetMemHandle into the current device
address space. For contexts on different devices cuIpcOpenMemHandle can attempt to enable peer
access between the devices as if the user called cuCtxEnablePeerAccess. This behavior is controlled by
the CU_IPC_MEM_LAZY_ENABLE_PEER_ACCESS flag. cuDeviceCanAccessPeer can determine
if a mapping is possible.

Contexts that may open CUipcMemHandles are restricted in the following way. CUipcMemHandles
from each CUdevice in a given process may only be opened by one CUcontext per CUdevice per other
process.

If the memory handle has already been opened by the current context, the reference count on the handle
is incremented by 1 and the existing device pointer is returned.

Memory returned from cuIpcOpenMemHandle must be freed with cuIpcCloseMemHandle.

Calling cuMemFree on an exported memory region before calling cuIpcCloseMemHandle in the
importing context will result in undefined behavior.

IPC functionality is restricted to devices with support for unified addressing on Linux and Windows
operating systems. IPC functionality on Windows is supported for compatibility purposes but not
recommended as it comes with performance cost. Users can test their device for IPC functionality by
calling cuapiDeviceGetAttribute with CU_DEVICE_ATTRIBUTE_IPC_EVENT_SUPPORTED


CUDA Driver API TRM-06703-001 _vRelease Version  |  194


Modules


Note:

No guarantees are made about the address returned in *pdptr. In particular, multiple processes may
not receive the same address for the same handle.


See also:

cuMemAlloc, cuMemFree, cuIpcGetEventHandle, cuIpcOpenEventHandle, cuIpcGetMemHandle,
cuIpcCloseMemHandle, cuCtxEnablePeerAccess, cuDeviceCanAccessPeer, cudaIpcOpenMemHandle
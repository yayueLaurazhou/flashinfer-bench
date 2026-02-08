# __host__cudaError_t cudaIpcOpenMemHandle (void **devPtr, cudaIpcMemHandle_t handle, unsigned int flags)

Opens an interprocess memory handle exported from another process and returns a device pointer
usable in the local process.

##### Parameters

**devPtr**

  - Returned device pointer
**handle**

  - cudaIpcMemHandle to open
**flags**

  - Flags for this operation. Must be specified as cudaIpcMemLazyEnablePeerAccess

##### Returns

cudaSuccess, cudaErrorMapBufferObjectFailed, cudaErrorInvalidResourceHandle,
cudaErrorDeviceUninitialized, cudaErrorTooManyPeers, cudaErrorNotSupported,
cudaErrorInvalidValue

##### Description

Maps memory exported from another process with cudaIpcGetMemHandle into the current device
address space. For contexts on different devices cudaIpcOpenMemHandle can attempt to enable
peer access between the devices as if the user called cudaDeviceEnablePeerAccess. This behavior is
controlled by the cudaIpcMemLazyEnablePeerAccess flag. cudaDeviceCanAccessPeer can determine
if a mapping is possible.

cudaIpcOpenMemHandle can open handles to devices that may not be visible in the process calling the
API.

Contexts that may open cudaIpcMemHandles are restricted in the following way. cudaIpcMemHandles
from each device in a given process may only be opened by one context per device per other process.

If the memory handle has already been opened by the current context, the reference count on the handle
is incremented by 1 and the existing device pointer is returned.

Memory returned from cudaIpcOpenMemHandle must be freed with cudaIpcCloseMemHandle.

Calling cudaFree on an exported memory region before calling cudaIpcCloseMemHandle in the
importing context will result in undefined behavior.

IPC functionality is restricted to devices with support for unified addressing on Linux and Windows
operating systems. IPC functionality on Windows is supported for compatibility purposes but not
recommended as it comes with performance cost. Users can test their device for IPC functionality by
calling cudaDeviceGetAttribute with cudaDevAttrIpcEventSupport


CUDA Runtime API vRelease Version  |  39


Modules


See also:

cudaMalloc, cudaFree, cudaIpcGetEventHandle, cudaIpcOpenEventHandle, cudaIpcGetMemHandle,
cudaIpcCloseMemHandle, cudaDeviceEnablePeerAccess, cudaDeviceCanAccessPeer,
cuIpcOpenMemHandle
# CUresult cuIpcOpenEventHandle (CUevent *phEvent, CUipcEventHandle handle)

Opens an interprocess event handle for use in the current process.

###### Parameters

**phEvent**

  - Returns the imported event
**handle**

  - Interprocess handle to open

###### Returns

CUDA_SUCCESS, CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_MAP_FAILED,
CUDA_ERROR_PEER_ACCESS_UNSUPPORTED, CUDA_ERROR_INVALID_HANDLE,
CUDA_ERROR_INVALID_VALUE

###### Description

Opens an interprocess event handle exported from another process with cuIpcGetEventHandle.
This function returns a CUevent that behaves like a locally created event with the
CU_EVENT_DISABLE_TIMING flag specified. This event must be freed with cuEventDestroy.

Performing operations on the imported event after the exported event has been freed with
cuEventDestroy will result in undefined behavior.

IPC functionality is restricted to devices with support for unified addressing on Linux and Windows
operating systems. IPC functionality on Windows is supported for compatibility purposes but not
recommended as it comes with performance cost. Users can test their device for IPC functionality by
calling cuapiDeviceGetAttribute with CU_DEVICE_ATTRIBUTE_IPC_EVENT_SUPPORTED


See also:

cuEventCreate, cuEventDestroy, cuEventSynchronize, cuEventQuery, cuStreamWaitEvent,
cuIpcGetEventHandle, cuIpcGetMemHandle, cuIpcOpenMemHandle, cuIpcCloseMemHandle,
cudaIpcOpenEventHandle


CUDA Driver API TRM-06703-001 _vRelease Version  |  193


Modules
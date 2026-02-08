# CUresult cuIpcGetEventHandle (CUipcEventHandle *pHandle, CUevent event)

Gets an interprocess handle for a previously allocated event.

###### Parameters

**pHandle**

  - Pointer to a user allocated CUipcEventHandle in which to return the opaque event handle
**event**

  - Event allocated with CU_EVENT_INTERPROCESS and CU_EVENT_DISABLE_TIMING
flags.

###### Returns

CUDA_SUCCESS, CUDA_ERROR_INVALID_HANDLE, CUDA_ERROR_OUT_OF_MEMORY,
CUDA_ERROR_MAP_FAILED, CUDA_ERROR_INVALID_VALUE

###### Description

Takes as input a previously allocated event. This event must have been created with the
CU_EVENT_INTERPROCESS and CU_EVENT_DISABLE_TIMING flags set. This opaque
handle may be copied into other processes and opened with cuIpcOpenEventHandle to allow efficient
hardware synchronization between GPU work in different processes.

After the event has been opened in the importing process, cuEventRecord, cuEventSynchronize,
cuStreamWaitEvent and cuEventQuery may be used in either process. Performing operations on the
imported event after the exported event has been freed with cuEventDestroy will result in undefined
behavior.


CUDA Driver API TRM-06703-001 _vRelease Version  |  191


Modules


IPC functionality is restricted to devices with support for unified addressing on Linux and Windows
operating systems. IPC functionality on Windows is supported for compatibility purposes but not
recommended as it comes with performance cost. Users can test their device for IPC functionality by
calling cuDeviceGetAttribute with CU_DEVICE_ATTRIBUTE_IPC_EVENT_SUPPORTED


See also:

cuEventCreate, cuEventDestroy, cuEventSynchronize, cuEventQuery, cuStreamWaitEvent,
cuIpcOpenEventHandle, cuIpcGetMemHandle, cuIpcOpenMemHandle, cuIpcCloseMemHandle,
cudaIpcGetEventHandle
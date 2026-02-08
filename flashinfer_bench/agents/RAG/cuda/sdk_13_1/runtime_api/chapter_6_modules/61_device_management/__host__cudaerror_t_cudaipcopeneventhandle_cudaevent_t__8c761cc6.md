# __host__cudaError_t cudaIpcOpenEventHandle (cudaEvent_t *event, cudaIpcEventHandle_t handle)

Opens an interprocess event handle for use in the current process.

##### Parameters

**event**

  - Returns the imported event
**handle**

  - Interprocess handle to open

##### Returns

cudaSuccess, cudaErrorMapBufferObjectFailed, cudaErrorNotSupported, cudaErrorInvalidValue,
cudaErrorDeviceUninitialized

##### Description

Opens an interprocess event handle exported from another process with cudaIpcGetEventHandle.
This function returns a cudaEvent_t that behaves like a locally created event with the
cudaEventDisableTiming flag specified. This event must be freed with cudaEventDestroy.

Performing operations on the imported event after the exported event has been freed with
cudaEventDestroy will result in undefined behavior.

IPC functionality is restricted to devices with support for unified addressing on Linux and Windows
operating systems. IPC functionality on Windows is supported for compatibility purposes but not
recommended as it comes with performance cost. Users can test their device for IPC functionality by
calling cudaDeviceGetAttribute with cudaDevAttrIpcEventSupport





See also:

cudaEventCreate, cudaEventDestroy, cudaEventSynchronize, cudaEventQuery,
cudaStreamWaitEvent, cudaIpcGetEventHandle, cudaIpcGetMemHandle, cudaIpcOpenMemHandle,
cudaIpcCloseMemHandle, cuIpcOpenEventHandle


CUDA Runtime API vRelease Version  |  38


Modules
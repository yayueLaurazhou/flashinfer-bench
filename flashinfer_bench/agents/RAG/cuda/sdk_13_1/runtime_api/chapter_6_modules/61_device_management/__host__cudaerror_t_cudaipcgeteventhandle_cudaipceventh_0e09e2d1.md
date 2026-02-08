# __host__cudaError_t cudaIpcGetEventHandle (cudaIpcEventHandle_t *handle, cudaEvent_t event)

Gets an interprocess handle for a previously allocated event.

##### Parameters

**handle**

  - Pointer to a user allocated cudaIpcEventHandle in which to return the opaque event handle
**event**

  - Event allocated with cudaEventInterprocess and cudaEventDisableTiming flags.

##### Returns

cudaSuccess, cudaErrorInvalidResourceHandle, cudaErrorMemoryAllocation,
cudaErrorMapBufferObjectFailed, cudaErrorNotSupported, cudaErrorInvalidValue

##### Description

Takes as input a previously allocated event. This event must have been created with the
cudaEventInterprocess and cudaEventDisableTiming flags set. This opaque handle may be copied into
other processes and opened with cudaIpcOpenEventHandle to allow efficient hardware synchronization
between GPU work in different processes.

After the event has been been opened in the importing process, cudaEventRecord,
cudaEventSynchronize, cudaStreamWaitEvent and cudaEventQuery may be used in either
process. Performing operations on the imported event after the exported event has been freed with
cudaEventDestroy will result in undefined behavior.

IPC functionality is restricted to devices with support for unified addressing on Linux and Windows
operating systems. IPC functionality on Windows is supported for compatibility purposes but not
recommended as it comes with performance cost. Users can test their device for IPC functionality by
calling cudaDeviceGetAttribute with cudaDevAttrIpcEventSupport





See also:

cudaEventCreate, cudaEventDestroy, cudaEventSynchronize, cudaEventQuery,
cudaStreamWaitEvent, cudaIpcOpenEventHandle, cudaIpcGetMemHandle, cudaIpcOpenMemHandle,
cudaIpcCloseMemHandle, cuIpcGetEventHandle


CUDA Runtime API vRelease Version  |  36


Modules
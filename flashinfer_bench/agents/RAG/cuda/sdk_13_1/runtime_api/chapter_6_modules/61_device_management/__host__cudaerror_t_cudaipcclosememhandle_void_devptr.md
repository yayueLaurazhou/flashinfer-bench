# __host__cudaError_t cudaIpcCloseMemHandle (void *devPtr)

Attempts to close memory mapped with cudaIpcOpenMemHandle.

##### Parameters

**devPtr**

  - Device pointer returned by cudaIpcOpenMemHandle

##### Returns

cudaSuccess, cudaErrorMapBufferObjectFailed, cudaErrorNotSupported, cudaErrorInvalidValue

##### Description

Decrements the reference count of the memory returnd by cudaIpcOpenMemHandle by 1. When
the reference count reaches 0, this API unmaps the memory. The original allocation in the exporting
process as well as imported mappings in other processes will be unaffected.

Any resources used to enable peer access will be freed if this is the last mapping using them.

IPC functionality is restricted to devices with support for unified addressing on Linux and Windows
operating systems. IPC functionality on Windows is supported for compatibility purposes but not
recommended as it comes with performance cost. Users can test their device for IPC functionality by
calling cudaDeviceGetAttribute with cudaDevAttrIpcEventSupport


See also:

cudaMalloc, cudaFree, cudaIpcGetEventHandle, cudaIpcOpenEventHandle, cudaIpcGetMemHandle,
cudaIpcOpenMemHandle, cuIpcCloseMemHandle


CUDA Runtime API vRelease Version  |  35


Modules
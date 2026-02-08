# __host__cudaError_t cudaIpcGetMemHandle (cudaIpcMemHandle_t *handle, void *devPtr)

Gets an interprocess memory handle for an existing device memory allocation.

##### Parameters

**handle**

  - Pointer to user allocated cudaIpcMemHandle to return the handle in.
**devPtr**

  - Base pointer to previously allocated device memory

##### Returns

cudaSuccess, cudaErrorMemoryAllocation, cudaErrorMapBufferObjectFailed,
cudaErrorNotSupported, cudaErrorInvalidValue

##### Description

Takes a pointer to the base of an existing device memory allocation created with cudaMalloc and
exports it for use in another process. This is a lightweight operation and may be called multiple times
on an allocation without adverse effects.

If a region of memory is freed with cudaFree and a subsequent call to cudaMalloc returns memory with
the same device address, cudaIpcGetMemHandle will return a unique handle for the new memory.

IPC functionality is restricted to devices with support for unified addressing on Linux and Windows
operating systems. IPC functionality on Windows is supported for compatibility purposes but not
recommended as it comes with performance cost. Users can test their device for IPC functionality by
calling cudaDeviceGetAttribute with cudaDevAttrIpcEventSupport





See also:

cudaMalloc, cudaFree, cudaIpcGetEventHandle, cudaIpcOpenEventHandle, cudaIpcOpenMemHandle,
cudaIpcCloseMemHandle, cuIpcGetMemHandle


CUDA Runtime API vRelease Version  |  37


Modules
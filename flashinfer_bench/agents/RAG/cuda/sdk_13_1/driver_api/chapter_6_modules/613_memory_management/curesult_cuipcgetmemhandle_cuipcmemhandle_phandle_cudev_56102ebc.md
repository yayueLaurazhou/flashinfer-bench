# CUresult cuIpcGetMemHandle (CUipcMemHandle *pHandle, CUdeviceptr dptr)

Gets an interprocess memory handle for an existing device memory allocation.

###### Parameters

**pHandle**

  - Pointer to user allocated CUipcMemHandle to return the handle in.
**dptr**

  - Base pointer to previously allocated device memory

###### Returns

CUDA_SUCCESS, CUDA_ERROR_INVALID_HANDLE, CUDA_ERROR_INVALID_CONTEXT,
CUDA_ERROR_OUT_OF_MEMORY, CUDA_ERROR_MAP_FAILED,
CUDA_ERROR_INVALID_VALUE

###### Description

Takes a pointer to the base of an existing device memory allocation created with cuMemAlloc and
exports it for use in another process. This is a lightweight operation and may be called multiple times
on an allocation without adverse effects.

If a region of memory is freed with cuMemFree and a subsequent call to cuMemAlloc returns memory
with the same device address, cuIpcGetMemHandle will return a unique handle for the new memory.

IPC functionality is restricted to devices with support for unified addressing on Linux and Windows
operating systems. IPC functionality on Windows is supported for compatibility purposes but not
recommended as it comes with performance cost. Users can test their device for IPC functionality by
calling cuapiDeviceGetAttribute with CU_DEVICE_ATTRIBUTE_IPC_EVENT_SUPPORTED


See also:

cuMemAlloc, cuMemFree, cuIpcGetEventHandle, cuIpcOpenEventHandle, cuIpcOpenMemHandle,
cuIpcCloseMemHandle, cudaIpcGetMemHandle


CUDA Driver API TRM-06703-001 _vRelease Version  |  192


Modules
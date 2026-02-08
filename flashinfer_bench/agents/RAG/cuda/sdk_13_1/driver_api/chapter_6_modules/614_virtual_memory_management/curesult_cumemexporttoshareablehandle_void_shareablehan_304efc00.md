# CUresult cuMemExportToShareableHandle (void *shareableHandle, CUmemGenericAllocationHandle handle, CUmemAllocationHandleType handleType, unsigned long long flags)

Exports an allocation to a requested shareable handle type.

###### Parameters

**shareableHandle**

  - Pointer to the location in which to store the requested handle type
**handle**

  - CUDA handle for the memory allocation


CUDA Driver API TRM-06703-001 _vRelease Version  |  274


Modules


**handleType**

  - Type of shareable handle requested (defines type and size of the shareableHandle output
parameter)
**flags**

  - Reserved, must be zero

###### Returns

CUDA_SUCCESS, CUDA_ERROR_INVALID_VALUE, CUDA_ERROR_NOT_INITIALIZED,
CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_PERMITTED,
CUDA_ERROR_NOT_SUPPORTED

###### Description

Given a CUDA memory handle, create a shareable memory allocation handle that can be used to share
the memory with other processes. The recipient process can convert the shareable handle back into a
CUDA memory handle using cuMemImportFromShareableHandle and map it with cuMemMap. The
implementation of what this handle is and how it can be transferred is defined by the requested handle
type in handleType

Once all shareable handles are closed and the allocation is released, the allocated memory referenced
will be released back to the OS and uses of the CUDA handle afterward will lead to undefined
behavior.

This API can also be used in conjunction with other APIs (e.g. Vulkan, OpenGL) that support
importing memory from the shareable type


See also:

cuMemImportFromShareableHandle
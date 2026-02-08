# CUresult cuPointerSetAttribute (const void *value, CUpointer_attribute attribute, CUdeviceptr ptr)

Set attributes on a previously allocated memory region.

###### Parameters

**value**

  - Pointer to memory containing the value to be set
**attribute**

  - Pointer attribute to set
**ptr**

  - Pointer to a memory region allocated using CUDA memory allocation APIs

###### Returns

CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE,
CUDA_ERROR_INVALID_DEVICE

###### Description

The supported attributes are:

CU_POINTER_ATTRIBUTE_SYNC_MEMOPS:

###### **‣**

A boolean attribute that can either be set (1) or unset (0). When set, the region of memory that ptr
points to is guaranteed to always synchronize memory operations that are synchronous. If there are
some previously initiated synchronous memory operations that are pending when this attribute is set,
the function does not return until those memory operations are complete. See further documentation in
the section titled "API synchronization behavior" to learn more about cases when synchronous memory
operations can exhibit asynchronous behavior. value will be considered as a pointer to an unsigned
integer to which this attribute is to be set.


Note:


Note that this function may also return error codes from previous, asynchronous launches.


See also:


CUDA Driver API TRM-06703-001 _vRelease Version  |  330


Modules


cuPointerGetAttribute, cuPointerGetAttributes, cuMemAlloc, cuMemFree, cuMemAllocHost,
cuMemFreeHost, cuMemHostAlloc, cuMemHostRegister, cuMemHostUnregister
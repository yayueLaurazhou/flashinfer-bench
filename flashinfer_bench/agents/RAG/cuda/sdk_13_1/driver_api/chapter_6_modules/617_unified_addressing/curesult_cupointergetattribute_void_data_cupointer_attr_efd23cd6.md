# CUresult cuPointerGetAttribute (void *data, CUpointer_attribute attribute, CUdeviceptr ptr)

Returns information about a pointer.

###### Parameters

**data**

  - Returned pointer attribute value
**attribute**

  - Pointer attribute to query
**ptr**

  - Pointer

###### Returns

CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE,
CUDA_ERROR_INVALID_DEVICE

###### Description

The supported attributes are:

CU_POINTER_ATTRIBUTE_CONTEXT:

###### **‣**

Returns in *data the CUcontext in which ptr was allocated or registered. The type of data must be
CUcontext *.


CUDA Driver API TRM-06703-001 _vRelease Version  |  325


Modules


If ptr was not allocated by, mapped by, or registered with a CUcontext which uses unified virtual
addressing then CUDA_ERROR_INVALID_VALUE is returned.

CU_POINTER_ATTRIBUTE_MEMORY_TYPE:

###### **‣**

Returns in *data the physical memory type of the memory that ptr addresses as a CUmemorytype
enumerated value. The type of data must be unsigned int.

If ptr addresses device memory then *data is set to CU_MEMORYTYPE_DEVICE. The
particular CUdevice on which the memory resides is the CUdevice of the CUcontext returned by the
CU_POINTER_ATTRIBUTE_CONTEXT attribute of ptr.

If ptr addresses host memory then *data is set to CU_MEMORYTYPE_HOST.

If ptr was not allocated by, mapped by, or registered with a CUcontext which uses unified virtual
addressing then CUDA_ERROR_INVALID_VALUE is returned.

If the current CUcontext does not support unified virtual addressing then
CUDA_ERROR_INVALID_CONTEXT is returned.

CU_POINTER_ATTRIBUTE_DEVICE_POINTER:

###### **‣**

Returns in *data the device pointer value through which ptr may be accessed by kernels running in
the current CUcontext. The type of data must be CUdeviceptr *.

If there exists no device pointer value through which kernels running in the current CUcontext may
access ptr then CUDA_ERROR_INVALID_VALUE is returned.

If there is no current CUcontext then CUDA_ERROR_INVALID_CONTEXT is returned.

Except in the exceptional disjoint addressing cases discussed below, the value returned in *data will
equal the input value ptr.

CU_POINTER_ATTRIBUTE_HOST_POINTER:

###### **‣**

Returns in *data the host pointer value through which ptr may be accessed by by the host program.
The type of data must be void **. If there exists no host pointer value through which the host
program may directly access ptr then CUDA_ERROR_INVALID_VALUE is returned.

Except in the exceptional disjoint addressing cases discussed below, the value returned in *data will
equal the input value ptr.

CU_POINTER_ATTRIBUTE_P2P_TOKENS:

###### **‣**

Returns in *data two tokens for use with the nv-p2p.h Linux kernel interface. data must be a struct
of type CUDA_POINTER_ATTRIBUTE_P2P_TOKENS.

ptr must be a pointer to memory obtained from :cuMemAlloc(). Note that p2pToken and
vaSpaceToken are only valid for the lifetime of the source allocation. A subsequent allocation at the
same address may return completely different tokens. Querying this attribute has a side effect of setting
the attribute CU_POINTER_ATTRIBUTE_SYNC_MEMOPS for the region of memory that ptr
points to.

CU_POINTER_ATTRIBUTE_SYNC_MEMOPS:

###### **‣**

CUDA Driver API TRM-06703-001 _vRelease Version  |  326


Modules


A boolean attribute which when set, ensures that synchronous memory operations initiated on the
region of memory that ptr points to will always synchronize. See further documentation in the section
titled "API synchronization behavior" to learn more about cases when synchronous memory operations
can exhibit asynchronous behavior.

CU_POINTER_ATTRIBUTE_BUFFER_ID:

###### **‣**

Returns in *data a buffer ID which is guaranteed to be unique within the process. data must point to
an unsigned long long.

ptr must be a pointer to memory obtained from a CUDA memory allocation API. Every memory
allocation from any of the CUDA memory allocation APIs will have a unique ID over a process
lifetime. Subsequent allocations do not reuse IDs from previous freed allocations. IDs are only unique
within a single process.

CU_POINTER_ATTRIBUTE_IS_MANAGED:

###### **‣**

Returns in *data a boolean that indicates whether the pointer points to managed memory or not.

If ptr is not a valid CUDA pointer then CUDA_ERROR_INVALID_VALUE is returned.

CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL:

###### **‣**

Returns in *data an integer representing a device ordinal of a device against which the memory was
allocated or registered.

CU_POINTER_ATTRIBUTE_IS_LEGACY_CUDA_IPC_CAPABLE:

###### **‣**

Returns in *data a boolean that indicates if this pointer maps to an allocation that is suitable for
cudaIpcGetMemHandle.

CU_POINTER_ATTRIBUTE_RANGE_START_ADDR:

###### **‣**

Returns in *data the starting address for the allocation referenced by the device pointer ptr. Note
that this is not necessarily the address of the mapped region, but the address of the mappable address
range ptr references (e.g. from cuMemAddressReserve).

CU_POINTER_ATTRIBUTE_RANGE_SIZE:

###### **‣**

Returns in *data the size for the allocation referenced by the device pointer ptr. Note that this
is not necessarily the size of the mapped region, but the size of the mappable address range ptr
references (e.g. from cuMemAddressReserve). To retrieve the size of the mapped region, see
cuMemGetAddressRange

CU_POINTER_ATTRIBUTE_MAPPED:

###### **‣**

Returns in *data a boolean that indicates if this pointer is in a valid address range that is mapped to a
backing allocation.

CU_POINTER_ATTRIBUTE_ALLOWED_HANDLE_TYPES:

###### **‣**

Returns a bitmask of the allowed handle types for an allocation that may be passed to
cuMemExportToShareableHandle.


CUDA Driver API TRM-06703-001 _vRelease Version  |  327


Modules


CU_POINTER_ATTRIBUTE_MEMPOOL_HANDLE:

###### **‣**

Returns in *data the handle to the mempool that the allocation was obtained from.

CU_POINTER_ATTRIBUTE_IS_HW_DECOMPRESS_CAPABLE:

###### **‣**

Returns in *data a boolean that indicates whether the pointer points to memory that is capable to be
used for hardware accelerated decompression.


Note that for most allocations in the unified virtual address space the host and device pointer for
accessing the allocation will be the same. The exceptions to this are

user memory registered using cuMemHostRegister

###### **‣**

host memory allocated using cuMemHostAlloc with the

###### **‣**

CU_MEMHOSTALLOC_WRITECOMBINED flag For these types of allocation there will exist
separate, disjoint host and device addresses for accessing the allocation. In particular
The host address will correspond to an invalid unmapped device address (which will result in an

###### **‣**

exception if accessed from the device)
The device address will correspond to an invalid unmapped host address

###### **‣**

(which will result in an exception if accessed from the host). For these types
of allocations, querying CU_POINTER_ATTRIBUTE_HOST_POINTER and
CU_POINTER_ATTRIBUTE_DEVICE_POINTER may be used to retrieve the host and device
addresses from either address.


Note:


Note that this function may also return error codes from previous, asynchronous launches.


See also:

cuPointerSetAttribute, cuMemAlloc, cuMemFree, cuMemAllocHost, cuMemFreeHost,
cuMemHostAlloc, cuMemHostRegister, cuMemHostUnregister, cudaPointerGetAttributes
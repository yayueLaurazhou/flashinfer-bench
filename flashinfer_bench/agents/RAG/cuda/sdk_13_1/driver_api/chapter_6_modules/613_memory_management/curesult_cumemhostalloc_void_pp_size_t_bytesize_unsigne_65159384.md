# CUresult cuMemHostAlloc (void **pp, size_t bytesize, unsigned int Flags)

Allocates page-locked host memory.

###### Parameters

**pp**

  - Returned pointer to host memory


CUDA Driver API TRM-06703-001 _vRelease Version  |  244


Modules


**bytesize**

  - Requested allocation size in bytes
**Flags**

  - Flags for allocation request

###### Returns

CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE,
CUDA_ERROR_OUT_OF_MEMORY

###### Description

Allocates bytesize bytes of host memory that is page-locked and accessible to the device. The
driver tracks the virtual memory ranges allocated with this function and automatically accelerates calls
to functions such as cuMemcpyHtoD(). Since the memory can be accessed directly by the device, it can
be read or written with much higher bandwidth than pageable memory obtained with functions such as
malloc().

On systems where
CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS_USES_HOST_PAGE_TABLES is
true, cuMemHostAlloc may not page-lock the allocated memory.

Page-locking excessive amounts of memory may degrade system performance, since it reduces the
amount of memory available to the system for paging. As a result, this function is best used sparingly
to allocate staging areas for data exchange between host and device.

The Flags parameter enables different options to be specified that affect the allocation, as follows.

CU_MEMHOSTALLOC_PORTABLE: The memory returned by this call will be considered as

###### **‣**

pinned memory by all CUDA contexts, not just the one that performed the allocation.

CU_MEMHOSTALLOC_DEVICEMAP: Maps the allocation into the CUDA address space. The

###### **‣**

device pointer to the memory may be obtained by calling cuMemHostGetDevicePointer().

CU_MEMHOSTALLOC_WRITECOMBINED: Allocates the memory as write-combined

###### **‣**

(WC). WC memory can be transferred across the PCI Express bus more quickly on some system
configurations, but cannot be read efficiently by most CPUs. WC memory is a good option for
buffers that will be written by the CPU and read by the GPU via mapped pinned memory or host>device transfers.

All of these flags are orthogonal to one another: a developer may allocate memory that is portable,
mapped and/or write-combined with no restrictions.

The CU_MEMHOSTALLOC_DEVICEMAP flag may be specified on CUDA contexts
for devices that do not support mapped pinned memory. The failure is deferred to
cuMemHostGetDevicePointer() because the memory may be mapped into other CUDA contexts via the
CU_MEMHOSTALLOC_PORTABLE flag.

The memory allocated by this function must be freed with cuMemFreeHost().


CUDA Driver API TRM-06703-001 _vRelease Version  |  245


Modules


Note all host memory allocated using cuMemHostAlloc() will automatically be immediately
accessible to all contexts on all devices which support unified addressing (as may be
queried using CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING). Unless the flag
CU_MEMHOSTALLOC_WRITECOMBINED is specified, the device pointer that may be
used to access this host memory from those contexts is always equal to the returned host pointer
*pp. If the flag CU_MEMHOSTALLOC_WRITECOMBINED is specified, then the function
cuMemHostGetDevicePointer() must be used to query the device pointer, even if the context supports
unified addressing. See Unified Addressing for additional details.


Note:


Note that this function may also return error codes from previous, asynchronous launches.


See also:

cuArray3DCreate, cuArray3DGetDescriptor, cuArrayCreate, cuArrayDestroy, cuArrayGetDescriptor,
cuMemAlloc, cuMemAllocHost, cuMemAllocPitch, cuMemcpy2D, cuMemcpy2DAsync,
cuMemcpy2DUnaligned, cuMemcpy3D, cuMemcpy3DAsync, cuMemcpyAtoA, cuMemcpyAtoD,
cuMemcpyAtoH, cuMemcpyAtoHAsync, cuMemcpyDtoA, cuMemcpyDtoD, cuMemcpyDtoDAsync,
cuMemcpyDtoH, cuMemcpyDtoHAsync, cuMemcpyHtoA, cuMemcpyHtoAAsync, cuMemcpyHtoD,
cuMemcpyHtoDAsync, cuMemFree, cuMemFreeHost, cuMemGetAddressRange, cuMemGetInfo,
cuMemHostGetDevicePointer, cuMemsetD2D8, cuMemsetD2D16, cuMemsetD2D32, cuMemsetD8,
cuMemsetD16, cuMemsetD32, cudaHostAlloc
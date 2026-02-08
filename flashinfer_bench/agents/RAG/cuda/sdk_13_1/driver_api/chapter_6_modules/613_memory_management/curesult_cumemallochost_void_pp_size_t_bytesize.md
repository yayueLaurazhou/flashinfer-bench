# CUresult cuMemAllocHost (void **pp, size_t bytesize)

Allocates page-locked host memory.

###### Parameters

**pp**

  - Returned pointer to host memory
**bytesize**

  - Requested allocation size in bytes

###### Returns

CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE,
CUDA_ERROR_OUT_OF_MEMORY

###### Description

Allocates bytesize bytes of host memory that is page-locked and accessible to the device. The
driver tracks the virtual memory ranges allocated with this function and automatically accelerates calls
to functions such as cuMemcpy(). Since the memory can be accessed directly by the device, it can be
read or written with much higher bandwidth than pageable memory obtained with functions such as
malloc().

On systems where
CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS_USES_HOST_PAGE_TABLES is
true, cuMemAllocHost may not page-lock the allocated memory.

Page-locking excessive amounts of memory with cuMemAllocHost() may degrade system
performance, since it reduces the amount of memory available to the system for paging. As a result,
this function is best used sparingly to allocate staging areas for data exchange between host and device.

Note all host memory allocated using cuMemAllocHost() will automatically be immediately
accessible to all contexts on all devices which support unified addressing (as may be queried using
CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING). The device pointer that may be used to
access this host memory from those contexts is always equal to the returned host pointer *pp. See
Unified Addressing for additional details.


Note:


Note that this function may also return error codes from previous, asynchronous launches.


See also:


CUDA Driver API TRM-06703-001 _vRelease Version  |  196


Modules


cuArray3DCreate, cuArray3DGetDescriptor, cuArrayCreate, cuArrayDestroy, cuArrayGetDescriptor,
cuMemAlloc, cuMemAllocPitch, cuMemcpy2D, cuMemcpy2DAsync, cuMemcpy2DUnaligned,
cuMemcpy3D, cuMemcpy3DAsync, cuMemcpyAtoA, cuMemcpyAtoD, cuMemcpyAtoH,
cuMemcpyAtoHAsync, cuMemcpyDtoA, cuMemcpyDtoD, cuMemcpyDtoDAsync, cuMemcpyDtoH,
cuMemcpyDtoHAsync, cuMemcpyHtoA, cuMemcpyHtoAAsync, cuMemcpyHtoD,
cuMemcpyHtoDAsync, cuMemFree, cuMemFreeHost, cuMemGetAddressRange, cuMemGetInfo,
cuMemHostAlloc, cuMemHostGetDevicePointer, cuMemsetD2D8, cuMemsetD2D16,
cuMemsetD2D32, cuMemsetD8, cuMemsetD16, cuMemsetD32, cudaMallocHost
# CUresult cuMemHostRegister (void *p, size_t bytesize, unsigned int Flags)

Registers an existing host memory range for use by CUDA.

###### Parameters

**p**

  - Host pointer to memory to page-lock
**bytesize**

  - Size in bytes of the address range to page-lock
**Flags**

  - Flags for allocation request


CUDA Driver API TRM-06703-001 _vRelease Version  |  248


Modules

###### Returns

CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE,
CUDA_ERROR_OUT_OF_MEMORY,
CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED,
CUDA_ERROR_NOT_PERMITTED, CUDA_ERROR_NOT_SUPPORTED

###### Description

Page-locks the memory range specified by p and bytesize and maps it for the device(s) as specified
by Flags. This memory range also is added to the same tracking mechanism as cuMemHostAlloc
to automatically accelerate calls to functions such as cuMemcpyHtoD(). Since the memory can be
accessed directly by the device, it can be read or written with much higher bandwidth than pageable
memory that has not been registered. Page-locking excessive amounts of memory may degrade system
performance, since it reduces the amount of memory available to the system for paging. As a result,
this function is best used sparingly to register staging areas for data exchange between host and device.

On systems where
CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS_USES_HOST_PAGE_TABLES
is true, cuMemHostRegister will not page-lock the memory range specified by ptr but only populate
unpopulated pages.

The Flags parameter enables different options to be specified that affect the allocation, as follows.

CU_MEMHOSTREGISTER_PORTABLE: The memory returned by this call will be considered as

###### **‣**

pinned memory by all CUDA contexts, not just the one that performed the allocation.

CU_MEMHOSTREGISTER_DEVICEMAP: Maps the allocation into the CUDA address space.

###### **‣**

The device pointer to the memory may be obtained by calling cuMemHostGetDevicePointer().

CU_MEMHOSTREGISTER_IOMEMORY: The pointer is treated as pointing to some I/O

###### **‣**

memory space, e.g. the PCI Express resource of a 3rd party device.

CU_MEMHOSTREGISTER_READ_ONLY: The pointer is treated as pointing

###### **‣**

to memory that is considered read-only by the device. On platforms without
CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS_USES_HOST_PAGE_TABLES,
this flag is required in order to register memory mapped to the CPU as readonly. Support for the use of this flag can be queried from the device attribute
CU_DEVICE_ATTRIBUTE_READ_ONLY_HOST_REGISTER_SUPPORTED. Using this
flag with a current context associated with a device that does not have this attribute set will cause
cuMemHostRegister to error with CUDA_ERROR_NOT_SUPPORTED.

All of these flags are orthogonal to one another: a developer may page-lock memory that is portable or
mapped with no restrictions.

The CU_MEMHOSTREGISTER_DEVICEMAP flag may be specified on CUDA
contexts for devices that do not support mapped pinned memory. The failure is deferred to


CUDA Driver API TRM-06703-001 _vRelease Version  |  249


Modules


cuMemHostGetDevicePointer() because the memory may be mapped into other CUDA contexts via the
CU_MEMHOSTREGISTER_PORTABLE flag.

For devices that have a non-zero value for the device attribute
CU_DEVICE_ATTRIBUTE_CAN_USE_HOST_POINTER_FOR_REGISTERED_MEM, the
memory can also be accessed from the device using the host pointer p. The device pointer returned
by cuMemHostGetDevicePointer() may or may not match the original host pointer ptr and depends
on the devices visible to the application. If all devices visible to the application have a non-zero value
for the device attribute, the device pointer returned by cuMemHostGetDevicePointer() will match the
original pointer ptr. If any device visible to the application has a zero value for the device attribute,
the device pointer returned by cuMemHostGetDevicePointer() will not match the original host pointer
ptr, but it will be suitable for use on all devices provided Unified Virtual Addressing is enabled. In
such systems, it is valid to access the memory using either pointer on devices that have a non-zero
value for the device attribute. Note however that such devices should access the memory using only of
the two pointers and not both.

The memory page-locked by this function must be unregistered with cuMemHostUnregister().


Note:


Note that this function may also return error codes from previous, asynchronous launches.


See also:

cuMemHostUnregister, cuMemHostGetFlags, cuMemHostGetDevicePointer, cudaHostRegister
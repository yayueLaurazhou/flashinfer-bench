# __host__cudaError_t cudaHostRegister (void *ptr, size_t size, unsigned int flags)

Registers an existing host memory range for use by CUDA.

##### Parameters

**ptr**

  - Host pointer to memory to page-lock
**size**

  - Size in bytes of the address range to page-lock in bytes
**flags**

  - Flags for allocation request

##### Returns

cudaSuccess, cudaErrorInvalidValue, cudaErrorMemoryAllocation,
cudaErrorHostMemoryAlreadyRegistered, cudaErrorNotSupported

##### Description

Page-locks the memory range specified by ptr and size and maps it for the device(s) as specified
by flags. This memory range also is added to the same tracking mechanism as cudaHostAlloc() to
automatically accelerate calls to functions such as cudaMemcpy(). Since the memory can be accessed
directly by the device, it can be read or written with much higher bandwidth than pageable memory that
has not been registered. Page-locking excessive amounts of memory may degrade system performance,


CUDA Runtime API vRelease Version  |  131


Modules


since it reduces the amount of memory available to the system for paging. As a result, this function is
best used sparingly to register staging areas for data exchange between host and device.

On systems where pageableMemoryAccessUsesHostPageTables is true, cudaHostRegister will not
page-lock the memory range specified by ptr but only populate unpopulated pages.

cudaHostRegister is supported only on I/O coherent devices that have a non-zero value for the device
attribute cudaDevAttrHostRegisterSupported.

The flags parameter enables different options to be specified that affect the allocation, as follows.

cudaHostRegisterDefault: On a system with unified virtual addressing, the memory will be both

##### **‣**

mapped and portable. On a system with no unified virtual addressing, the memory will be neither
mapped nor portable.

cudaHostRegisterPortable: The memory returned by this call will be considered as pinned memory

##### **‣**

by all CUDA contexts, not just the one that performed the allocation.

cudaHostRegisterMapped: Maps the allocation into the CUDA address space. The device pointer to

##### **‣**

the memory may be obtained by calling cudaHostGetDevicePointer().

cudaHostRegisterIoMemory: The passed memory pointer is treated as pointing to some memory##### **‣**
mapped I/O space, e.g. belonging to a third-party PCIe device, and it will marked as non cachecoherent and contiguous.

cudaHostRegisterReadOnly: The passed memory pointer is treated as pointing

##### **‣**

to memory that is considered read-only by the device. On platforms without
cudaDevAttrPageableMemoryAccessUsesHostPageTables, this flag is required in order to register
memory mapped to the CPU as read-only. Support for the use of this flag can be queried from
the device attribute cudaDevAttrHostRegisterReadOnlySupported. Using this flag with a current
context associated with a device that does not have this attribute set will cause cudaHostRegister to
error with cudaErrorNotSupported.

All of these flags are orthogonal to one another: a developer may page-lock memory that is portable or
mapped with no restrictions.

The CUDA context must have been created with the cudaMapHost flag in order for the
cudaHostRegisterMapped flag to have any effect.

The cudaHostRegisterMapped flag may be specified on CUDA contexts for devices that do not support
mapped pinned memory. The failure is deferred to cudaHostGetDevicePointer() because the memory
may be mapped into other CUDA contexts via the cudaHostRegisterPortable flag.

For devices that have a non-zero value for the device attribute
cudaDevAttrCanUseHostPointerForRegisteredMem, the memory can also be accessed from the device
using the host pointer ptr. The device pointer returned by cudaHostGetDevicePointer() may or
may not match the original host pointer ptr and depends on the devices visible to the application.
If all devices visible to the application have a non-zero value for the device attribute, the device
pointer returned by cudaHostGetDevicePointer() will match the original pointer ptr. If any device
visible to the application has a zero value for the device attribute, the device pointer returned by


CUDA Runtime API vRelease Version  |  132


Modules


cudaHostGetDevicePointer() will not match the original host pointer ptr, but it will be suitable for
use on all devices provided Unified Virtual Addressing is enabled. In such systems, it is valid to access
the memory using either pointer on devices that have a non-zero value for the device attribute. Note
however that such devices should access the memory using only of the two pointers and not both.

The memory page-locked by this function must be unregistered with cudaHostUnregister().





See also:

cudaHostUnregister, cudaHostGetFlags, cudaHostGetDevicePointer, cuMemHostRegister
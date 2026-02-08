# __host__cudaError_t cudaMallocHost (void **ptr, size_t size, unsigned int flags)

[C++ API] Allocates page-locked memory on the host

##### Parameters

**ptr**

  - Device pointer to allocated memory
**size**

  - Requested allocation size in bytes


CUDA Runtime API vRelease Version  |  489


Modules


**flags**

  - Requested properties of allocated memory

##### Returns

cudaSuccess, cudaErrorMemoryAllocation

##### Description

Allocates size bytes of host memory that is page-locked and accessible to the device. The driver
tracks the virtual memory ranges allocated with this function and automatically accelerates calls to
functions such as cudaMemcpy(). Since the memory can be accessed directly by the device, it can be
read or written with much higher bandwidth than pageable memory obtained with functions such as
malloc(). Allocating excessive amounts of pinned memory may degrade system performance, since it
reduces the amount of memory available to the system for paging. As a result, this function is best used
sparingly to allocate staging areas for data exchange between host and device.

The flags parameter enables different options to be specified that affect the allocation, as follows.

cudaHostAllocDefault: This flag's value is defined to be 0.

##### **‣**

cudaHostAllocPortable: The memory returned by this call will be considered as pinned memory by

##### **‣**

all CUDA contexts, not just the one that performed the allocation.
cudaHostAllocMapped: Maps the allocation into the CUDA address space. The device pointer to

##### **‣**

the memory may be obtained by calling cudaHostGetDevicePointer().
cudaHostAllocWriteCombined: Allocates the memory as write-combined (WC). WC memory can

##### **‣**

be transferred across the PCI Express bus more quickly on some system configurations, but cannot
be read efficiently by most CPUs. WC memory is a good option for buffers that will be written by
the CPU and read by the device via mapped pinned memory or host->device transfers.

All of these flags are orthogonal to one another: a developer may allocate memory that is portable,
mapped and/or write-combined with no restrictions.

cudaSetDeviceFlags() must have been called with the cudaDeviceMapHost flag in order for the
cudaHostAllocMapped flag to have any effect.

The cudaHostAllocMapped flag may be specified on CUDA contexts for devices that do not support
mapped pinned memory. The failure is deferred to cudaHostGetDevicePointer() because the memory
may be mapped into other CUDA contexts via the cudaHostAllocPortable flag.

Memory allocated by this function must be freed with cudaFreeHost().





CUDA Runtime API vRelease Version  |  490


Modules


See also:

cudaSetDeviceFlags, cudaMallocHost ( C API), cudaFreeHost, cudaHostAlloc
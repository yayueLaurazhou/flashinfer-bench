# 6.13. Unified Addressing

This section describes the unified addressing functions of the CUDA runtime application programming
interface.

Overview

CUDA devices can share a unified address space with the host. For these devices there is no distinction
between a device pointer and a host pointer -- the same pointer value may be used to access memory
from the host program and from a kernel running on the device (with exceptions enumerated below).

Supported Platforms

Whether or not a device supports unified addressing may be queried by calling
cudaGetDeviceProperties() with the device property cudaDeviceProp::unifiedAddressing.

Unified addressing is automatically enabled in 64-bit processes .

Looking Up Information from Pointer Values

It is possible to look up information about the memory which backs a pointer value. For instance, one
may want to know if a pointer points to host or device memory. As another example, in the case of
device memory, one may want to know on which CUDA device the memory resides. These properties
may be queried using the function cudaPointerGetAttributes()

Since pointers are unique, it is not necessary to specify information about the pointers specified to
cudaMemcpy() and other copy functions. The copy direction cudaMemcpyDefault may be used to
specify that the CUDA runtime should infer the location of the pointer from its value.

Automatic Mapping of Host Allocated Host Memory

All host memory allocated through all devices using cudaMallocHost() and cudaHostAlloc() is always
directly accessible from all devices that support unified addressing. This is the case regardless of
whether or not the flags cudaHostAllocPortable and cudaHostAllocMapped are specified.

The pointer value through which allocated host memory may be accessed in kernels on all devices that
support unified addressing is the same as the pointer value through which that memory is accessed
on the host. It is not necessary to call cudaHostGetDevicePointer() to get the device pointer for these
allocations.

Note that this is not the case for memory allocated using the flag cudaHostAllocWriteCombined, as
discussed below.

Direct Access of Peer Memory

Upon enabling direct access from a device that supports unified addressing to another peer device that
supports unified addressing using cudaDeviceEnablePeerAccess() all memory allocated in the peer


CUDA Runtime API vRelease Version  |  226


Modules


device using cudaMalloc() and cudaMallocPitch() will immediately be accessible by the current device.
The device pointer value through which any peer's memory may be accessed in the current device is the
same pointer value through which that memory may be accessed from the peer device.

Exceptions, Disjoint Addressing

Not all memory may be accessed on devices through the same pointer value through which they are
accessed on the host. These exceptions are host memory registered using cudaHostRegister() and host
memory allocated using the flag cudaHostAllocWriteCombined. For these exceptions, there exists a
distinct host and device address for the memory. The device address is guaranteed to not overlap any
valid host pointer range and is guaranteed to have the same value across all devices that support unified
addressing.

This device address may be queried using cudaHostGetDevicePointer() when a device using unified
addressing is current. Either the host or the unified device pointer value may be used to refer to this
memory in cudaMemcpy() and similar functions using the cudaMemcpyDefault memory direction.
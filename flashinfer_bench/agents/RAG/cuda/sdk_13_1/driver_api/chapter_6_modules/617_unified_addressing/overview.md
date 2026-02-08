# 6.17. Unified Addressing

This section describes the unified addressing functions of the low-level CUDA driver application
programming interface.

Overview

CUDA devices can share a unified address space with the host. For these devices there is no distinction
between a device pointer and a host pointer -- the same pointer value may be used to access memory
from the host program and from a kernel running on the device (with exceptions enumerated below).

Supported Platforms

Whether or not a device supports unified addressing may be queried by calling cuDeviceGetAttribute()
with the device attribute CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING.

Unified addressing is automatically enabled in 64-bit processes

Looking Up Information from Pointer Values

It is possible to look up information about the memory which backs a pointer value. For instance, one
may want to know if a pointer points to host or device memory. As another example, in the case of
device memory, one may want to know on which CUDA device the memory resides. These properties
may be queried using the function cuPointerGetAttribute()

Since pointers are unique, it is not necessary to specify information about the pointers specified to
the various copy functions in the CUDA API. The function cuMemcpy() may be used to perform
a copy between two pointers, ignoring whether they point to host or device memory (making
cuMemcpyHtoD(), cuMemcpyDtoD(), and cuMemcpyDtoH() unnecessary for devices supporting
unified addressing). For multidimensional copies, the memory type CU_MEMORYTYPE_UNIFIED
may be used to specify that the CUDA driver should infer the location of the pointer from its value.

Automatic Mapping of Host Allocated Host Memory


CUDA Driver API TRM-06703-001 _vRelease Version  |  311


Modules


All host memory allocated in all contexts using cuMemAllocHost() and cuMemHostAlloc() is
always directly accessible from all contexts on all devices that support unified addressing. This
is the case regardless of whether or not the flags CU_MEMHOSTALLOC_PORTABLE and
CU_MEMHOSTALLOC_DEVICEMAP are specified.

The pointer value through which allocated host memory may be accessed in kernels on all devices that
support unified addressing is the same as the pointer value through which that memory is accessed on
the host, so it is not necessary to call cuMemHostGetDevicePointer() to get the device pointer for these
allocations.

Note that this is not the case for memory allocated using the flag
CU_MEMHOSTALLOC_WRITECOMBINED, as discussed below.

Automatic Registration of Peer Memory

Upon enabling direct access from a context that supports unified addressing to another peer context that
supports unified addressing using cuCtxEnablePeerAccess() all memory allocated in the peer context
using cuMemAlloc() and cuMemAllocPitch() will immediately be accessible by the current context.
The device pointer value through which any peer memory may be accessed in the current context is the
same pointer value through which that memory may be accessed in the peer context.

Exceptions, Disjoint Addressing

Not all memory may be accessed on devices through the same pointer value through which they are
accessed on the host. These exceptions are host memory registered using cuMemHostRegister() and
host memory allocated using the flag CU_MEMHOSTALLOC_WRITECOMBINED. For these
exceptions, there exists a distinct host and device address for the memory. The device address is
guaranteed to not overlap any valid host pointer range and is guaranteed to have the same value across
all contexts that support unified addressing.

This device address may be queried using cuMemHostGetDevicePointer() when a context using
unified addressing is current. Either the host or the unified device pointer value may be used to refer
to this memory through cuMemcpy() and similar functions using the CU_MEMORYTYPE_UNIFIED
memory type.
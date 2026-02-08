# CUresult cuMulticastCreate (CUmemGenericAllocationHandle *mcHandle, const CUmulticastObjectProp *prop)

Create a generic allocation handle representing a multicast object described by the given properties.

###### Parameters

**mcHandle**
Value of handle returned.
**prop**
Properties of the multicast object to create.

###### Returns

CUDA_SUCCESS, CUDA_ERROR_INVALID_VALUE, CUDA_ERROR_OUT_OF_MEMORY,
CUDA_ERROR_INVALID_DEVICE, CUDA_ERROR_NOT_INITIALIZED,
CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_PERMITTED,
CUDA_ERROR_NOT_SUPPORTED


CUDA Driver API TRM-06703-001 _vRelease Version  |  308


Modules

###### Description

This creates a multicast object as described by prop. The number of participating devices is
specified by CUmulticastObjectProp::numDevices. Devices can be added to the multicast object
via cuMulticastAddDevice. All participating devices must be added to the multicast object before
memory can be bound to it. Memory is bound to the multicast object via cuMulticastBindMem,
cuMulticastBindMem_v2, cuMulticastBindAddr, or cuMulticastBindAddr_v2. and can be
unbound via cuMulticastUnbind. The total amount of memory that can be bound per device is
specified by :CUmulticastObjectProp::size. This size must be a multiple of the value returned by
cuMulticastGetGranularity with the flag CU_MULTICAST_GRANULARITY_MINIMUM. For best
performance however, the size should be aligned to the value returned by cuMulticastGetGranularity
with the flag CU_MULTICAST_GRANULARITY_RECOMMENDED.

After all participating devices have been added, multicast objects can also be mapped to a
device's virtual address space using the virtual memory management APIs (see cuMemMap and
cuMemSetAccess). Multicast objects can also be shared with other processes by requesting a shareable
handle via cuMemExportToShareableHandle. Note that the desired types of shareable handles must be
specified in the bitmask CUmulticastObjectProp::handleTypes. Multicast objects can be released using
the virtual memory management API cuMemRelease.


See also:

cuMulticastAddDevice, cuMulticastBindMem, cuMulticastBindAddr, cuMulticastUnbind

cuMemCreate, cuMemRelease, cuMemExportToShareableHandle,
cuMemImportFromShareableHandle

cuMulticastBindAddr_v2, cuMulticastBindMem_v2
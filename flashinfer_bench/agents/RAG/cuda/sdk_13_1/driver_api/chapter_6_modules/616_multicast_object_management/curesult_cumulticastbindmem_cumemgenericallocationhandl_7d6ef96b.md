# CUresult cuMulticastBindMem (CUmemGenericAllocationHandle mcHandle, size_t mcOffset, CUmemGenericAllocationHandle memHandle, size_t memOffset, size_t size, unsigned long long flags)

Bind a memory allocation represented by a handle to a multicast object.

###### Parameters

**mcHandle**
Handle representing a multicast object.


CUDA Driver API TRM-06703-001 _vRelease Version  |  305


Modules


**mcOffset**
Offset into the multicast object for attachment.
**memHandle**
Handle representing a memory allocation.
**memOffset**
Offset into the memory for attachment.
**size**
Size of the memory that will be bound to the multicast object.
**flags**
Flags for future use, must be zero for now.

###### Returns

CUDA_SUCCESS, CUDA_ERROR_INVALID_VALUE, CUDA_ERROR_INVALID_DEVICE,
CUDA_ERROR_NOT_INITIALIZED, CUDA_ERROR_DEINITIALIZED,
CUDA_ERROR_NOT_PERMITTED, CUDA_ERROR_NOT_SUPPORTED,
CUDA_ERROR_OUT_OF_MEMORY, CUDA_ERROR_SYSTEM_NOT_READY,
CUDA_ERROR_ILLEGAL_STATE

###### Description

Binds a memory allocation specified by memHandle and created via cuMemCreate to a
multicast object represented by mcHandle and created via cuMulticastCreate. The intended
size of the bind, the offset in the multicast range mcOffset as well as the offset in the
memory memOffset must be a multiple of the value returned by cuMulticastGetGranularity
with the flag CU_MULTICAST_GRANULARITY_MINIMUM. For best performance
however, size, mcOffset and memOffset should be aligned to the granularity of
the memory allocation(see ::cuMemGetAllocationGranularity) or to the value returned by
cuMulticastGetGranularity with the flag CU_MULTICAST_GRANULARITY_RECOMMENDED.

The size + memOffset cannot be larger than the size of the allocated memory. Similarly
the size + mcOffset cannot be larger than the size of the multicast object. The memory
allocation must have beeen created on one of the devices that was added to the multicast team via
cuMulticastAddDevice. Externally shareable as well as imported multicast objects can be bound only
to externally shareable memory. Note that this call will return CUDA_ERROR_OUT_OF_MEMORY
if there are insufficient resources required to perform the bind. This call may also return
CUDA_ERROR_SYSTEM_NOT_READY if the necessary system software is not initialized or
running.

This call may return CUDA_ERROR_ILLEGAL_STATE if the system configuration is in an illegal
state. In such cases, to continue using multicast, verify that the system configuration is in a valid state
and all required driver daemons are running properly.


See also:

cuMulticastCreate, cuMulticastAddDevice, cuMemCreate


CUDA Driver API TRM-06703-001 _vRelease Version  |  306


Modules


cuMulticastBindMem_v2
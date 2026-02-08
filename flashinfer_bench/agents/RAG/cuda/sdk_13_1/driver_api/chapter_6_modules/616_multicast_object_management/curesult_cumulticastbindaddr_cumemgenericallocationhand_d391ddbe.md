# CUresult cuMulticastBindAddr (CUmemGenericAllocationHandle mcHandle, size_t mcOffset, CUdeviceptr memptr, size_t size, unsigned long long flags)

Bind a memory allocation represented by a virtual address to a multicast object.

###### Parameters

**mcHandle**
Handle representing a multicast object.
**mcOffset**
Offset into multicast va range for attachment.
**memptr**
Virtual address of the memory allocation.
**size**
Size of memory that will be bound to the multicast object.
**flags**
Flags for future use, must be zero now.

###### Returns

CUDA_SUCCESS, CUDA_ERROR_INVALID_VALUE, CUDA_ERROR_INVALID_DEVICE,
CUDA_ERROR_NOT_INITIALIZED, CUDA_ERROR_DEINITIALIZED,
CUDA_ERROR_NOT_PERMITTED, CUDA_ERROR_NOT_SUPPORTED,
CUDA_ERROR_OUT_OF_MEMORY, CUDA_ERROR_SYSTEM_NOT_READY,
CUDA_ERROR_ILLEGAL_STATE

###### Description

Binds a memory allocation specified by its mapped address memptr to a multicast object
represented by mcHandle. The memory must have been allocated via cuMemCreate or
cudaMallocAsync. The intended size of the bind, the offset in the multicast range mcOffset
and memptr must be a multiple of the value returned by cuMulticastGetGranularity with the flag
CU_MULTICAST_GRANULARITY_MINIMUM. For best performance however, size, mcOffset
and memptr should be aligned to the value returned by cuMulticastGetGranularity with the flag
CU_MULTICAST_GRANULARITY_RECOMMENDED.

The size cannot be larger than the size of the allocated memory. Similarly the size + mcOffset
cannot be larger than the total size of the multicast object. The memory allocation must have beeen
created on one of the devices that was added to the multicast team via cuMulticastAddDevice.
Externally shareable as well as imported multicast objects can be bound only to externally
shareable memory. Note that this call will return CUDA_ERROR_OUT_OF_MEMORY
if there are insufficient resources required to perform the bind. This call may also return


CUDA Driver API TRM-06703-001 _vRelease Version  |  303


Modules


CUDA_ERROR_SYSTEM_NOT_READY if the necessary system software is not initialized or
running.

This call may return CUDA_ERROR_ILLEGAL_STATE if the system configuration is in an illegal
state. In such cases, to continue using multicast, verify that the system configuration is in a valid state
and all required driver daemons are running properly.


See also:

cuMulticastCreate, cuMulticastAddDevice, cuMemCreate

cuMulticastBindAddr_v2
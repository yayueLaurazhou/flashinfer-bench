# CUresult cuMemMap (CUdeviceptr ptr, size_t size, size_t offset, CUmemGenericAllocationHandle handle, unsigned long long flags)

Maps an allocation handle to a reserved virtual address range.

###### Parameters

**ptr**

  - Address where memory will be mapped.
**size**

  - Size of the memory mapping.
**offset**
handle from which to start mapping Note: currently must be zero.

  - Offset into the memory represented by
**handle**

  - Handle to a shareable memory
**flags**

  - flags for future use, must be zero now.

###### Returns

CUDA_SUCCESS, CUDA_ERROR_INVALID_VALUE, CUDA_ERROR_INVALID_DEVICE,
CUDA_ERROR_OUT_OF_MEMORY, CUDA_ERROR_NOT_INITIALIZED,


CUDA Driver API TRM-06703-001 _vRelease Version  |  278


Modules


CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_PERMITTED,
CUDA_ERROR_NOT_SUPPORTED, CUDA_ERROR_ILLEGAL_STATE

from which to start mapping

###### ‣ handle

Note: currently must be zero.

###### **‣** Description

Maps bytes of memory represented by handle starting from byte offset to size to
address range [addr, addr + size]. This range must be an address reservation previously
reserved with cuMemAddressReserve, and offset + size must be less than the size of the
memory allocation. Both ptr, size, and offset must be a multiple of the value given via
cuMemGetAllocationGranularity with the CU_MEM_ALLOC_GRANULARITY_MINIMUM flag. If
handle represents a multicast object, ptr, size and offset must be aligned to the value returned
by cuMulticastGetGranularity with the flag CU_MULTICAST_MINIMUM_GRANULARITY.
For best performance however, it is recommended that ptr, size and offset
be aligned to the value returned by cuMulticastGetGranularity with the flag
CU_MULTICAST_RECOMMENDED_GRANULARITY.

When handle represents a multicast object, this call may return CUDA_ERROR_ILLEGAL_STATE
if the system configuration is in an illegal state. In such cases, to continue using multicast, verify that
the system configuration is in a valid state and all required driver daemons are running properly.

Please note calling cuMemMap does not make the address accessible, the caller needs to update
accessibility of a contiguous mapped VA range by calling cuMemSetAccess.

Once a recipient process obtains a shareable memory handle from
cuMemImportFromShareableHandle, the process must use cuMemMap to map the memory into its
address ranges before setting accessibility with cuMemSetAccess.

cuMemMap can only create mappings on VA range reservations that are not currently mapped.


Note:


Note that this function may also return error codes from previous, asynchronous launches.


See also:

cuMemUnmap, cuMemSetAccess, cuMemCreate, cuMemAddressReserve,
cuMemImportFromShareableHandle


CUDA Driver API TRM-06703-001 _vRelease Version  |  279


Modules
# CUresult cuMemAddressReserve (CUdeviceptr *ptr, size_t size, size_t alignment, CUdeviceptr addr, unsigned long long flags)

Allocate an address range reservation.

###### Parameters

**ptr**

  - Resulting pointer to start of virtual address range allocated
**size**

  - Size of the reserved virtual address range requested
**alignment**

  - Alignment of the reserved virtual address range requested
**addr**

  - Hint address for the start of the address range
**flags**

  - Currently unused, must be zero

###### Returns

CUDA_SUCCESS, CUDA_ERROR_INVALID_VALUE, CUDA_ERROR_OUT_OF_MEMORY,
CUDA_ERROR_NOT_INITIALIZED, CUDA_ERROR_DEINITIALIZED,
CUDA_ERROR_NOT_PERMITTED, CUDA_ERROR_NOT_SUPPORTED

###### Description

Reserves a virtual address range based on the given parameters, giving the starting address of the range
in ptr. This API requires a system that supports UVA. The size and address parameters must be a
multiple of the host page size and the alignment must be a power of two or zero for default alignment.
If addr is 0, then the driver chooses the address at which to place the start of the reservation whereas
when it is non-zero then the driver treats it as a hint about where to place the reservation.


See also:

cuMemAddressFree


CUDA Driver API TRM-06703-001 _vRelease Version  |  272


Modules
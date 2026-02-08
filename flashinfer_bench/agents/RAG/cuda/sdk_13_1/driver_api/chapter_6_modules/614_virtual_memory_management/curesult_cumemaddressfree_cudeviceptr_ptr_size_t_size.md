# CUresult cuMemAddressFree (CUdeviceptr ptr, size_t size)

Free an address range reservation.

###### Parameters

**ptr**

  - Starting address of the virtual address range to free
**size**

  - Size of the virtual address region to free

###### Returns

CUDA_SUCCESS, CUDA_ERROR_INVALID_VALUE, CUDA_ERROR_NOT_INITIALIZED,
CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_PERMITTED,
CUDA_ERROR_NOT_SUPPORTED

###### Description

Frees a virtual address range reserved by cuMemAddressReserve. The size must match what was given
to memAddressReserve and the ptr given must match what was returned from memAddressReserve.


See also:

cuMemAddressReserve


CUDA Driver API TRM-06703-001 _vRelease Version  |  271


Modules
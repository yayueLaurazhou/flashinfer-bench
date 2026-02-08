# CUresult cuMemUnmap (CUdeviceptr ptr, size_t size)

Unmap the backing memory of a given address range.

###### Parameters

**ptr**

  - Starting address for the virtual address range to unmap
**size**

  - Size of the virtual address range to unmap

###### Returns

CUDA_SUCCESS, CUDA_ERROR_INVALID_VALUE, CUDA_ERROR_NOT_INITIALIZED,
CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_PERMITTED,
CUDA_ERROR_NOT_SUPPORTED

###### Description

The range must be the entire contiguous address range that was mapped to. In other words,
cuMemUnmap cannot unmap a sub-range of an address range mapped by cuMemCreate / cuMemMap.


CUDA Driver API TRM-06703-001 _vRelease Version  |  285


Modules


Any backing memory allocations will be freed if there are no existing mappings and there are no
unreleased memory handles.

When cuMemUnmap returns successfully the address range is converted to an address reservation and
can be used for a future calls to cuMemMap. Any new mapping to this virtual address will need to have
access granted through cuMemSetAccess, as all mappings start with no accessibility setup.





See also:

cuMemCreate, cuMemAddressReserve
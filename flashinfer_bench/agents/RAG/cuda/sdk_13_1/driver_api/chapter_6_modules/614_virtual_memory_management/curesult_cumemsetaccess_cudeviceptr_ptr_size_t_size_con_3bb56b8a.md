# CUresult cuMemSetAccess (CUdeviceptr ptr, size_t size, const CUmemAccessDesc *desc, size_t count)

Set the access flags for each location specified in desc for the given virtual address range.

###### Parameters

**ptr**

  - Starting address for the virtual address range
**size**

  - Length of the virtual address range
**desc**
mapping for each location specified

  - Array of CUmemAccessDesc that describe how to change the
**count**

  - Number of CUmemAccessDesc in desc

###### Returns

CUDA_SUCCESS, CUDA_ERROR_INVALID_VALUE, CUDA_ERROR_INVALID_DEVICE,
CUDA_ERROR_NOT_SUPPORTED

mapping for each location specified

###### **‣**

CUDA Driver API TRM-06703-001 _vRelease Version  |  284


Modules

###### Description

Given the virtual address range via ptr and size, and the locations in the array given by desc
and count, set the access flags for the target locations. The range must be a fully mapped
address range containing all allocations created by cuMemMap / cuMemCreate. Users cannot
specify CU_MEM_LOCATION_TYPE_HOST_NUMA accessibility for allocations created
on with other location types. Note: When CUmemAccessDesc::CUmemLocation::type is
CU_MEM_LOCATION_TYPE_HOST_NUMA, CUmemAccessDesc::CUmemLocation::id is
ignored. When setting the access flags for a virtual address range mapping a multicast object,
ptr and size must be aligned to the value returned by cuMulticastGetGranularity with the flag
CU_MULTICAST_MINIMUM_GRANULARITY. For best performance however, it is recommended
that ptr and size be aligned to the value returned by cuMulticastGetGranularity with the flag
CU_MULTICAST_RECOMMENDED_GRANULARITY.





See also:

cuMemSetAccess, cuMemCreate, :cuMemMap
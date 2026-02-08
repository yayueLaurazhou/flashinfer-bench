# CUresult cuMemGetAllocationGranularity (size_t *granularity, const CUmemAllocationProp *prop, CUmemAllocationGranularity_flags option)

Calculates either the minimal or recommended granularity.

###### Parameters

**granularity**
Returned granularity.
**prop**
Property for which to determine the granularity for
**option**
Determines which granularity to return

###### Returns

CUDA_SUCCESS, CUDA_ERROR_INVALID_VALUE, CUDA_ERROR_NOT_INITIALIZED,
CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_PERMITTED,
CUDA_ERROR_NOT_SUPPORTED

###### Description

Calculates either the minimal or recommended granularity for a given allocation specification and
returns it in granularity. This granularity can be used as a multiple for alignment, size, or address
mapping.


See also:

cuMemCreate, cuMemMap


CUDA Driver API TRM-06703-001 _vRelease Version  |  276


Modules
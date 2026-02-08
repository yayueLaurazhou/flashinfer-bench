# CUresult cuMulticastGetGranularity (size_t *granularity, const CUmulticastObjectProp *prop, CUmulticastGranularity_flags option)

Calculates either the minimal or recommended granularity for multicast object.

###### Parameters

**granularity**
Returned granularity.
**prop**
Properties of the multicast object.
**option**
Determines which granularity to return.


CUDA Driver API TRM-06703-001 _vRelease Version  |  309


Modules

###### Returns

CUDA_SUCCESS, CUDA_ERROR_INVALID_VALUE, CUDA_ERROR_NOT_INITIALIZED,
CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_PERMITTED,
CUDA_ERROR_NOT_SUPPORTED

###### Description

Calculates either the minimal or recommended granularity for a given set of multicast object properties
and returns it in granularity. This granularity can be used as a multiple for size, bind offsets and address
mappings of the multicast object.


See also:

cuMulticastCreate, cuMulticastBindMem, cuMulticastBindAddr, cuMulticastUnbind

cuMulticastBindMem_v2, cuMulticastBindAddr_v2
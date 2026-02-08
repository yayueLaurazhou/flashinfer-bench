# CUresult cuMemGetAccess (unsigned long long *flags, const CUmemLocation *location, CUdeviceptr ptr)

Get the access flags set for the given location and ptr.

###### Parameters

**flags**

  - Flags set for this location
**location**

  - Location in which to check the flags for
**ptr**

  - Address in which to check the access flags for


CUDA Driver API TRM-06703-001 _vRelease Version  |  275


Modules

###### Returns

CUDA_SUCCESS, CUDA_ERROR_INVALID_VALUE, CUDA_ERROR_INVALID_DEVICE,
CUDA_ERROR_NOT_INITIALIZED, CUDA_ERROR_DEINITIALIZED,
CUDA_ERROR_NOT_PERMITTED, CUDA_ERROR_NOT_SUPPORTED

###### Description

See also:

cuMemSetAccess
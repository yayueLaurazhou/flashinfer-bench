# CUresult cuDeviceGetDevResource (CUdevice device, CUdevResource *resource, CUdevResourceType type)

Get device resources.

###### Parameters

**device**

  - Device to get resource for
**resource**

  - Output pointer to a CUdevResource structure
**type**

  - Type of resource to retrieve

###### Returns

CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
CUDA_ERROR_INVALID_RESOURCE_TYPE, CUDA_ERROR_INVALID_VALUE,
CUDA_ERROR_INVALID_DEVICE

###### Description

Get the type resources available to the device. This may often be the starting point for further
partitioning or configuring of resources.

Note: The API is not supported on 32-bit platforms.


CUDA Driver API TRM-06703-001 _vRelease Version  |  576


Modules


See also:

cuDevResourceGenerateDesc
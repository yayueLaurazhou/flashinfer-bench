# CUresult cuDevResourceGenerateDesc (CUdevResourceDesc *phDesc, CUdevResource *resources, unsigned int nbResources)

Generate a resource descriptor.

###### Parameters

**phDesc**

  - Output descriptor
**resources**

  - Array of resources to be included in the descriptor
**nbResources**

  - Number of resources passed in resources

###### Returns

CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
CUDA_ERROR_INVALID_VALUE, CUDA_ERROR_INVALID_RESOURCE_TYPE,
CUDA_ERROR_INVALID_RESOURCE_CONFIGURATION

###### Description

Generates a single resource descriptor with the set of resources specified in resources. The
generated resource descriptor is necessary for the creation of green contexts via the cuGreenCtxCreate
API. Resources of the same type can be passed in, provided they meet the requirements as noted below.

A successful API call must have:

A valid output pointer for the descriptor as well as a valid array of

###### ‣ phDesc resources

pointers, with the array size passed in nbResources. If multiple resources are
provided in resources, the device they came from must be the same, otherwise
CUDA_ERROR_INVALID_RESOURCE_CONFIGURATION is returned. If multiple
resources are provided in resources and they are of type CU_DEV_RESOURCE_TYPE_SM,
they must be outputs (whether result or remaining) from the same split
API instance and have the same smCoscheduledAlignment values, otherwise
CUDA_ERROR_INVALID_RESOURCE_CONFIGURATION is returned.

Note: The API is not supported on 32-bit platforms.


See also:

cuDevSmResourceSplitByCount


CUDA Driver API TRM-06703-001 _vRelease Version  |  577


Modules
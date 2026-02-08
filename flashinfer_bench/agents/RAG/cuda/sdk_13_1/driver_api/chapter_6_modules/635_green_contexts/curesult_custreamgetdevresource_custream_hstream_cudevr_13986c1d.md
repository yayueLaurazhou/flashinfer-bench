# CUresult cuStreamGetDevResource (CUstream hStream, CUdevResource *resource, CUdevResourceType type)

Get stream resources.

###### Parameters

**hStream**

  - Stream to get resource for


CUDA Driver API TRM-06703-001 _vRelease Version  |  588


Modules


**resource**

  - Output pointer to a CUdevResource structure
**type**

  - Type of resource to retrieve

###### Returns

CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
CUDA_ERROR_INVALID_RESOURCE_TYPE, CUDA_ERROR_INVALID_VALUE,
CUDA_ERROR_INVALID_HANDLE

###### Description

Get the type resources available to the hStream and store them in resource.

Note: The API will return CUDA_ERROR_INVALID_RESOURCE_TYPE
is type is CU_DEV_RESOURCE_TYPE_WORKQUEUE_CONFIG or
CU_DEV_RESOURCE_TYPE_WORKQUEUE.


Note:


Note that this function may also return error codes from previous, asynchronous launches.


See also:

cuGreenCtxCreate, cuGreenCtxStreamCreate, cuStreamCreate, cuDevSmResourceSplitByCount,
cuDevResourceGenerateDesc, cudaStreamGetDevResource
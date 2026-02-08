# CUresult cuGreenCtxGetDevResource (CUgreenCtx hCtx, CUdevResource *resource, CUdevResourceType type)

Get green context resources.

###### Parameters

**hCtx**

  - Green context to get resource for
**resource**

  - Output pointer to a CUdevResource structure
**type**

  - Type of resource to retrieve


CUDA Driver API TRM-06703-001 _vRelease Version  |  584


Modules

###### Returns

CUDA_SUCCESS CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_RESOURCE_TYPE,
CUDA_ERROR_INVALID_VALUE

###### Description

Get the type resources available to the green context represented by hCtx


See also:

cuDevResourceGenerateDesc
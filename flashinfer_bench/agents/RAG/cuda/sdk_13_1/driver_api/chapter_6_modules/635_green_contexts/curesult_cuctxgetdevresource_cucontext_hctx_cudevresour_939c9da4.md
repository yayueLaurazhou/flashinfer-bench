# CUresult cuCtxGetDevResource (CUcontext hCtx, CUdevResource *resource, CUdevResourceType type)

Get context resources.

###### Parameters

**hCtx**

  - Context to get resource for


CUDA Driver API TRM-06703-001 _vRelease Version  |  575


Modules


**resource**

  - Output pointer to a CUdevResource structure
**type**

  - Type of resource to retrieve

###### Returns

CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
CUDA_ERROR_INVALID_RESOURCE_TYPE, CUDA_ERROR_INVALID_VALUE,
CUDA_ERROR_INVALID_CONTEXT

###### Description

Get the type resources available to the context represented by hCtx Note: The API is not supported
on 32-bit platforms.


See also:

cuDevResourceGenerateDesc
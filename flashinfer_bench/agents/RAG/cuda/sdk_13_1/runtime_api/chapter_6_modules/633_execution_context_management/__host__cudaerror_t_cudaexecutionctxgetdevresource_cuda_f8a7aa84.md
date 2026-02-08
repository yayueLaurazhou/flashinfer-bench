# __host__cudaError_t cudaExecutionCtxGetDevResource (cudaExecutionContext_t ctx, cudaDevResource *resource, cudaDevResourceType type)

Get context resources.

##### Parameters

**ctx**

  - Execution context to get resource for (required parameter, see note below)
**resource**

  - Output pointer to a cudaDevResource structure
**type**

  - Type of resource to retrieve

##### Returns

cudaSuccess, cudaErrorInvalidValue, cudaErrorNotSupported, cudaErrorNotPermitted,
cudaErrorCudartUnloading, cudaErrorInitializationError

##### Description

Get the type resources available to context represented by ctx.

Note: The API is not supported on 32-bit platforms.



See also:

cudaDeviceGetDevResource, cudaDevSmResourceSplit, cudaDevResourceGenerateDesc,
cudaGreenCtxCreate


CUDA Runtime API vRelease Version  |  454


Modules
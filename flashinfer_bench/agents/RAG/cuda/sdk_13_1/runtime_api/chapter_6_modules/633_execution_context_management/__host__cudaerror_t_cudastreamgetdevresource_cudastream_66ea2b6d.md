# __host__cudaError_t cudaStreamGetDevResource (cudaStream_t hStream, cudaDevResource *resource, cudaDevResourceType type)

Get stream resources.

##### Parameters

**hStream**

  - Stream to get resource for
**resource**

  - Output pointer to a cudaDevResource structure
**type**

  - Type of resource to retrieve

##### Returns

cudaSuccess, cudaErrorCudartUnloading, cudaErrorInitializationError, cudaErrorDeviceUninitialized,
cudaErrorInvalidResourceType, cudaErrorInvalidValue, cudaErrorInvalidHandle,
cudaErrorNotPermitted, cudaErrorCallRequiresNewerDriver,

##### Description

Get the type resources available to the hStream and store them in resource.

Note: The API will return cudaErrorInvalidResourceType is type is
cudaDevResourceTypeWorkqueueConfig or cudaDevResourceTypeWorkqueue.


CUDA Runtime API vRelease Version  |  461


Modules



See also:

cudaGreenCtxCreate, cudaExecutionCtxStreamCreate, cudaStreamCreate, cudaDevSmResourceSplit,
cudaDevResourceGenerateDesc, cuStreamGetDevResource
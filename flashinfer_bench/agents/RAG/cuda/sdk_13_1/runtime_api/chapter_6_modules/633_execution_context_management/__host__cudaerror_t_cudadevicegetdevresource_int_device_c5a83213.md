# __host__cudaError_t cudaDeviceGetDevResource (int device, cudaDevResource *resource, cudaDevResourceType type)

Get device resources.

##### Parameters

**device**

  - Device to get resource for
**resource**

  - Output pointer to a cudaDevResource structure
**type**

  - Type of resource to retrieve


CUDA Runtime API vRelease Version  |  444


Modules

##### Returns

cudaSuccess, cudaErrorInvalidValue, cudaErrorNotPermitted, cudaErrorInvalidDevice,
cudaErrorInvalidResourceType, cudaErrorNotSupported, cudaErrorCudartUnloading,
cudaErrorInitializationError

##### Description

Get the type resources available to the device. This may often be the starting point for further
partitioning or configuring of resources.

Note: The API is not supported on 32-bit platforms.





See also:

cuDeviceGetDevResource, cudaExecutionCtxGetDevResource, cudaDevSmResourceSplit,
cudaDevResourceGenerateDesc
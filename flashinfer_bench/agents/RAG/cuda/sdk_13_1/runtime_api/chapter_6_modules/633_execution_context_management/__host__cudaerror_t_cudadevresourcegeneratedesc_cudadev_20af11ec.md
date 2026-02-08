# __host__cudaError_t cudaDevResourceGenerateDesc (cudaDevResourceDesc_t *phDesc, cudaDevResource *resources, unsigned int nbResources)

Generate a resource descriptor.

##### Parameters

**phDesc**

  - Output descriptor
**resources**

  - Array of resources to be included in the descriptor
**nbResources**

  - Number of resources passed in resources

##### Returns

cudaSuccess, cudaErrorInvalidValue, cudaErrorNotPermitted, cudaErrorInvalidResourceType,
cudaErrorInvalidResourceConfiguration, cudaErrorNotSupported, cudaErrorOutOfMemory,
cudaErrorCudartUnloading, cudaErrorInitializationError

##### Description

Generates a single resource descriptor with the set of resources specified in resources.
The generated resource descriptor is necessary for the creation of green contexts via the
cudaGreenCtxCreate API. Resources of the same type can be passed in, provided they meet the
requirements as noted below.

A successful API call must have:

A valid output pointer for the descriptor as well as a valid array of pointers,

##### ‣ phDesc resources

with the array size passed in nbResources. If multiple resources are provided in resources,
the device they came from must be the same, otherwise cudaErrorInvalidResourceConfiguration
is returned. If multiple resources are provided in resources and they are of type
cudaDevResourceTypeSm, they must be outputs (whether result or remaining) from
the same split API instance and have the same smCoscheduledAlignment values, otherwise
cudaErrorInvalidResourceConfiguration is returned.

Note: The API is not supported on 32-bit platforms.


Note:


CUDA Runtime API vRelease Version  |  446


Modules


See also:

cuDevResourceGenerateDesc, cudaDeviceGetDevResource, cudaExecutionCtxGetDevResource,
cudaDevSmResourceSplit, cudaGreenCtxCreate
# __host__cudaError_t cudaGreenCtxCreate (cudaExecutionContext_t *phCtx, cudaDevResourceDesc_t desc, int device, unsigned int flags)

Creates a green context with a specified set of resources.

##### Parameters

**phCtx**

  - Pointer for the output handle to the green context
**desc**

  - Descriptor generated via cudaDevResourceGenerateDesc which contains the set of resources to be
used
**device**

  - Device on which to create the green context.
**flags**

  - Green context creation flags. Must be 0, currently reserved for future use.

##### Returns

cudaSuccess, cudaErrorInvalidValue, cudaErrorInvalidDevice, cudaErrorNotPermitted,
cudaErrorNotSupported, cudaErrorOutOfMemory, cudaErrorCudartUnloading,
cudaErrorInitializationError

##### Description

This API creates a green context with the resources specified in the descriptor desc and returns it in
the handle represented by phCtx.

This API retains the device’s primary context for the lifetime of the green context. The primary
context will be released when the green context is destroyed. To avoid the overhead of repeated
initialization and teardown, it is recommended to explicitly initialize the device's primary context ahead
of time using cudaInitDevice. This ensures that the primary context remains initialized throughout the
program’s lifetime, minimizing overhead during green context creation and destruction.


CUDA Runtime API vRelease Version  |  460


Modules


The API does not create a default stream for the green context. Developers are expected to create
streams explicitly using cudaExecutionCtxStreamCreate to submit work to the green context.

Note: The API is not supported on 32-bit platforms.





See also:

cudaDeviceGetDevResource, cudaDevSmResourceSplit, cudaDevResourceGenerateDesc,
cudaExecutionCtxGetDevResource, cudaExecutionCtxDestroy, cudaInitDevice,
cudaExecutionCtxStreamCreate
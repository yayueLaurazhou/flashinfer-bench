# __host__cudaError_t cudaExecutionCtxSynchronize (cudaExecutionContext_t ctx)

Block for the specified execution context's tasks to complete.

##### Parameters

**ctx**

  - Execution context to synchronize (required parameter, see note below)

##### Returns

cudaSuccess, cudaErrorCudartUnloading, cudaErrorDeviceUninitialized, cudaErrorInvalidValue

##### Description

Blocks until the specified execution context has completed all preceding requested tasks. If the
specified execution context is the device (primary) context obtained via cudaDeviceGetExecutionCtx,
green contexts that have been created on the device will also be synchronized.

The API returns an error if one of the preceding tasks failed.


Note:

**‣** Note that this function may also return error codes from previous, asynchronous launches.


CUDA Runtime API vRelease Version  |  458


Modules





See also:

cudaGreenCtxCreate, cudaExecutionCtxDestroy, cudaDeviceSynchronize, cuCtxSynchronize_v2
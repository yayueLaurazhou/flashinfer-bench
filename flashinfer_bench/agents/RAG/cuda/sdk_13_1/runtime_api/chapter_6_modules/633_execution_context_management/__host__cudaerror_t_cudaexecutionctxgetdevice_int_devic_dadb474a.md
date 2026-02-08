# __host__cudaError_t cudaExecutionCtxGetDevice (int *device, cudaExecutionContext_t ctx)

Returns the device handle for the execution context.

##### Parameters

**device**

  - Returned device handle for the specified execution context
**ctx**

  - Execution context for which to obtain the device (required parameter, see note below)

##### Returns

cudaSuccess, cudaErrorCudartUnloading, cudaErrorInitializationError, cudaErrorInvalidValue,
cudaErrorNotPermitted

##### Description

Returns in *device the handle of the specified execution context's device. The execution context
should not be NULL.



See also:

cudaGreenCtxCreate, cudaExecutionCtxDestroy, cuCtxGetDevice


CUDA Runtime API vRelease Version  |  453


Modules
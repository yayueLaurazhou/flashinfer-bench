# __host__cudaError_t cudaDeviceGetExecutionCtx (cudaExecutionContext_t *ctx, int device)

Returns the execution context for a device.

##### Parameters

**ctx**

  - Returns the device execution context
**device**

  - Device to get the execution context for

##### Returns

cudaSuccess, cudaErrorInvalidValue, cudaErrorInvalidDevice

##### Description

Returns in ctx the execution context for the specified device. This is the device's primary context. The
returned context can then be passed to APIs that take in a cudaExecutionContext_t enabling explicit
context-based programming without relying on thread-local state.

Passing the returned execution context to cudaExecutionCtxDestroy() is not allowed and will result in
undefined behavior.


CUDA Runtime API vRelease Version  |  445


Modules


See also:

cudaExecutionCtxGetDevice, cudaExecutionCtxGetId
# __host__cudaError_t cudaExecutionCtxDestroy (cudaExecutionContext_t ctx)

Destroy a execution context.

##### Parameters

**ctx**

  - Execution context to destroy (required parameter, see note below)

##### Returns

cudaSuccess, cudaErrorInvalidValue, cudaErrorNotPermitted, cudaErrorCudartUnloading,
cudaErrorInitializationError

##### Description

Destroys the specified execution context ctx. It is the responsibility of the caller to ensure that no API
call issues using ctx while cudaExecutionCtxDestroy() is executing or subsequently.

If ctx is a green context, any resources provisioned for it (that were initially available via the resource
descriptor) are released as well.

The API does not destroy streams created via cudaExecutionCtxStreamCreate. Users are expected
to destroy these streams explicitly using cudaStreamDestroy to avoid resource leaks. Once the
execution context is destroyed, any subsequent API calls involving these streams will return
cudaErrorStreamDetached with the exception of the following APIs:

cudaStreamDestroy. Note this is only supported on CUDA drivers 13.1 and above.

##### **‣**

Additionally, the API will invalidate all active captures on these streams.

Passing in a ctx that was not explicitly created via CUDA Runtime APIs is not allowed and will result
in undefined behavior.





See also:

cudaGreenCtxCreate


CUDA Runtime API vRelease Version  |  452


Modules
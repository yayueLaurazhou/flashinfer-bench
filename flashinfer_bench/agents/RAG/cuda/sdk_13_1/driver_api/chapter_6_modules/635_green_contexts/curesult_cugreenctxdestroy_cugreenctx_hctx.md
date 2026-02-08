# CUresult cuGreenCtxDestroy (CUgreenCtx hCtx)

Destroys a green context.

###### Parameters

**hCtx**

  - Green context to be destroyed

###### Returns

CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_CONTEXT_IS_DESTROYED

###### Description

Destroys the green context, releasing the primary context of the device that this green context
was created for. Any resources provisioned for this green context (that were initially available
via the resource descriptor) are released as well. The API does not destroy streams created
via cuGreenCtxStreamCreate, cuStreamCreate, or cuStreamCreateWithPriority. Users are
expected to destroy these streams explicitly using cuStreamDestroy to avoid resource leaks.
Once the green context is destroyed, any subsequent API calls involving these streams will return
CUDA_ERROR_STREAM_DETACHED with the exception of the following APIs:

cuStreamDestroy.

###### **‣**

Additionally, the API will invalidate all active captures on these streams.


See also:

cuGreenCtxCreate, cuCtxDestroy
# __host__cudaError_t cudaExecutionCtxGetId (cudaExecutionContext_t ctx, unsigned long long *ctxId)

Returns the unique Id associated with the execution context supplied.

##### Parameters

**ctx**

  - Context for which to obtain the Id (required parameter, see note below)
**ctxId**

  - Pointer to store the Id of the context

##### Returns

cudaSuccess, cudaErrorCudartUnloading, cudaErrorInitializationError, cudaErrorInvalidValue,
cudaErrorNotPermitted

##### Description

Returns in ctxId the unique Id which is associated with a given context. The Id is unique for the life
of the program for this instance of CUDA. The execution context should not be NULL.



See also:

cudaGreenCtxCreate, cudaExecutionCtxDestroy, cudaExecutionCtxGetDevice, cuCtxGetId
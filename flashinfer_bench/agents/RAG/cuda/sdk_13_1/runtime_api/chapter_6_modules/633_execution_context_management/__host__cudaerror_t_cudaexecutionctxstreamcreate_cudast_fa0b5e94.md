# __host__cudaError_t cudaExecutionCtxStreamCreate (cudaStream_t *phStream, cudaExecutionContext_t ctx, unsigned int flags, int priority)

Creates a stream and initializes it for the given execution context.

##### Parameters

**phStream**

  - Returned stream handle
**ctx**

  - Execution context to initialize the stream with (required parameter, see note below)
**flags**

  - Flags for stream creation
**priority**

  - Stream priority

##### Returns

cudaSuccess, cudaErrorInvalidValue, cudaErrorNotPermitted, cudaErrorOutOfMemory,
cudaErrorCudartUnloading, cudaErrorInitializationError

##### Description

The API creates a CUDA stream with the specified flags and priority, initializing it with
resources as defined at the time of creating the specified ctx. Additionally, the API also enables work
submitted to to the stream to be tracked under ctx.

The supported values for flags are:

cudaStreamDefault: Default stream creation flag. This would be cudaStreamNonBlocking for

##### **‣**

streams created on a green context.
cudaStreamNonBlocking: Specifies that work running in the created stream may run concurrently

##### **‣**

with work in stream 0 (the NULL stream), and that the created stream should perform no implicit
synchronization with stream 0

Specifying priority affects the scheduling priority of work in the stream. Priorities provide a hint
to preferentially run work with higher priority when possible, but do not preempt already-running work
or provide any other functional guarantee on execution order. priority follows a convention where
lower numbers represent higher priorities. '0' represents default priority. The range of meaningful
numerical priorities can be queried using cudaDeviceGetStreamPriorityRange. If the specified priority
is outside the numerical range returned by cudaDeviceGetStreamPriorityRange, it will automatically be
clamped to the lowest or the highest number in the range.


CUDA Runtime API vRelease Version  |  457


Modules









See also:

cudaStreamDestroy, cudaGreenCtxCreate, cudaDeviceGetStreamPriorityRange, cudaStreamGetFlags,
cudaStreamGetPriority, cudaStreamGetDevice, cudaStreamGetDevResource, cudaLaunchKernel,
cudaEventRecord, cudaStreamWaitEvent, cudaStreamQuery, cudaStreamSynchronize,
cudaStreamAddCallback
# __host__cudaError_t cudaExecutionCtxWaitEvent (cudaExecutionContext_t ctx, cudaEvent_t event)

Make an execution context wait on an event.

##### Parameters

**ctx**

  - Execution context to wait for (required parameter, see note below)
**event**

  - Event to wait on

##### Returns

cudaSuccess, cudaErrorCudartUnloading, cudaErrorInitializationError, cudaErrorInvalidHandle,
cudaErrorStreamCaptureUnsupported

##### Description

Makes all future work submitted to execution context ctx wait for all work captured in event.
The synchronization will be performed on the device and will not block the calling CPU thread. See
cudaExecutionCtxRecordEvent() for details on what is captured by an event. If the execution context
passed to ctx is the device (primary) context obtained via cudaDeviceGetExecutionCtx(), all green
contexts created on the device will wait for event as well.





Note:

**‣** Note that this function may also return error codes from previous, asynchronous launches.


CUDA Runtime API vRelease Version  |  459


Modules





See also:

cudaExecutionCtxRecordEvent, cudaStreamWaitEvent, cuCtxWaitEvent, cuGreenCtxWaitEvent
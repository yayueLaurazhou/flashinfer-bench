# __host__cudaError_t cudaExecutionCtxRecordEvent (cudaExecutionContext_t ctx, cudaEvent_t event)

Records an event for the specified execution context.

##### Parameters

**ctx**

  - Execution context to record event for (required parameter, see note below)


CUDA Runtime API vRelease Version  |  455


Modules


**event**

  - Event to record

##### Returns

cudaSuccess, cudaErrorCudartUnloading, cudaErrorInitializationError, cudaErrorInvalidHandle,
cudaErrorStreamCaptureUnsupported

##### Description

Captures in event all the activities of the execution context ctx at the time of this call. event and
ctx must be from the same CUDA device, otherwise cudaErrorInvalidHandle will be returned. Calls
such as cudaEventQuery() or cudaExecutionCtxWaitEvent() will then examine or wait for completion
of the work that was captured. Uses of ctx after this call do not modify event. If the execution
context passed to ctx is the device (primary) context obtained via cudaDeviceGetExecutionCtx(),
event will capture all the activities of the green contexts created on the device as well.







See also:

cudaEventRecord, cudaExecutionCtxWaitEvent, cuCtxRecordEvent, cuGreenCtxRecordEvent


CUDA Runtime API vRelease Version  |  456


Modules
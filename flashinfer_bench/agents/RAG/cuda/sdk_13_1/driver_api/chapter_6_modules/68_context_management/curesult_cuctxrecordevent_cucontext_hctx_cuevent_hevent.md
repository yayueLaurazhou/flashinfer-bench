# CUresult cuCtxRecordEvent (CUcontext hCtx, CUevent hEvent)

Records an event.

###### Parameters

**hCtx**

  - Context to record event for
**hEvent**

  - Event to record

###### Returns

CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_HANDLE,
CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED

###### Description

Captures in hEvent all the activities of the context hCtx at the time of this call. hEvent and hCtx
must be from the same CUDA context, otherwise CUDA_ERROR_INVALID_HANDLE will be
returned. Calls such as cuEventQuery() or cuCtxWaitEvent() will then examine or wait for completion
of the work that was captured. Uses of hCtx after this call do not modify hEvent. If the context
passed to hCtx is the primary context, hEvent will capture all the activities of the primary context
and its green contexts. If the context passed to hCtx is a context converted from green context via
cuCtxFromGreenCtx(), hEvent will capture only the activities of the green context.





See also:

cuCtxWaitEvent, cuGreenCtxRecordEvent, cuGreenCtxWaitEvent, cuEventRecord
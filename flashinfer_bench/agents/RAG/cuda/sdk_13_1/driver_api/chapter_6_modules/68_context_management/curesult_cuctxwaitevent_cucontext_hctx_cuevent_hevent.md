# CUresult cuCtxWaitEvent (CUcontext hCtx, CUevent hEvent)

Make a context wait on an event.

###### Parameters

**hCtx**

  - Context to wait
**hEvent**

  - Event to wait on

###### Returns

CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_HANDLE,
CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED

###### Description

Makes all future work submitted to context hCtx wait for all work captured in hEvent. The
synchronization will be performed on the device and will not block the calling CPU thread. See
cuCtxRecordEvent() for details on what is captured by an event. If the context passed to hCtx is the
primary context, the primary context and its green contexts will wait for hEvent. If the context passed
to hCtx is a context converted from green context via cuCtxFromGreenCtx(), the green context will
wait for hEvent.





See also:

cuCtxRecordEvent, cuGreenCtxRecordEvent, cuGreenCtxWaitEvent, cuStreamWaitEvent
# CUresult cuGreenCtxWaitEvent (CUgreenCtx hCtx, CUevent hEvent)

Make a green context wait on an event.

###### Parameters

**hCtx**

  - Green context to wait
**hEvent**

  - Event to wait on

###### Returns

CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_HANDLE,
CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED

###### Description

Makes all future work submitted to green context hCtx wait for all work captured in hEvent. The
synchronization will be performed on the device and will not block the calling CPU thread. See
cuGreenCtxRecordEvent() or cuEventRecord(), for details on what is captured by an event.





See also:

cuGreenCtxRecordEvent, cuStreamWaitEvent, cuCtxRecordEvent, cuCtxWaitEvent
# __host__cudaError_t cudaEventRecordWithFlags (cudaEvent_t event, cudaStream_t stream, unsigned int flags)

Records an event.

##### Parameters

**event**

  - Event to record
**stream**

  - Stream in which to record event
**flags**

  - Parameters for the operation(See above)

##### Returns

cudaSuccess, cudaErrorInvalidValue, cudaErrorInvalidResourceHandle, cudaErrorLaunchFailure

##### Description

Captures in event the contents of stream at the time of this call. event and stream must be
on the same CUDA context. Calls such as cudaEventQuery() or cudaStreamWaitEvent() will then
examine or wait for completion of the work that was captured. Uses of stream after this call do not
modify event. See note on default stream behavior for what is captured in the default case.

cudaEventRecordWithFlags() can be called multiple times on the same event and will overwrite the
previously captured state. Other APIs such as cudaStreamWaitEvent() use the most recently captured
state at the time of the API call, and are not affected by later calls to cudaEventRecordWithFlags().
Before the first call to cudaEventRecordWithFlags(), an event represents an empty set of work, so for
example cudaEventQuery() would return cudaSuccess.

flags include:

cudaEventRecordDefault: Default event creation flag.

##### **‣**

cudaEventRecordExternal: Event is captured in the graph as an external event node when

##### **‣**

performing stream capture.











CUDA Runtime API vRelease Version  |  79


Modules





See also:

cudaEventCreate ( C API), cudaEventCreateWithFlags, cudaEventQuery, cudaEventSynchronize,
cudaEventDestroy, cudaEventElapsedTime, cudaStreamWaitEvent, cudaEventRecord, cuEventRecord,
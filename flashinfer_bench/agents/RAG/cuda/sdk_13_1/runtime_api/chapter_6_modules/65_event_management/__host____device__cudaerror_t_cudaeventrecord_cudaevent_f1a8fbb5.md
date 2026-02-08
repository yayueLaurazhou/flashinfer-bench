# __host____device__cudaError_t cudaEventRecord (cudaEvent_t event, cudaStream_t stream)

Records an event.

##### Parameters

**event**

  - Event to record
**stream**

  - Stream in which to record event

##### Returns

cudaSuccess, cudaErrorInvalidValue, cudaErrorInvalidResourceHandle, cudaErrorLaunchFailure

##### Description

Captures in event the contents of stream at the time of this call. event and stream must be
on the same CUDA context. Calls such as cudaEventQuery() or cudaStreamWaitEvent() will then
examine or wait for completion of the work that was captured. Uses of stream after this call do not
modify event. See note on default stream behavior for what is captured in the default case.

cudaEventRecord() can be called multiple times on the same event and will overwrite the previously
captured state. Other APIs such as cudaStreamWaitEvent() use the most recently captured state at the
time of the API call, and are not affected by later calls to cudaEventRecord(). Before the first call to
cudaEventRecord(), an event represents an empty set of work, so for example cudaEventQuery() would
return cudaSuccess.











See also:

cudaEventCreate ( C API), cudaEventCreateWithFlags, cudaEventQuery, cudaEventSynchronize,
cudaEventDestroy, cudaEventElapsedTime, cudaStreamWaitEvent, cudaEventRecordWithFlags,
cuEventRecord


CUDA Runtime API vRelease Version  |  78


Modules
# __host__cudaError_t cudaEventElapsedTime (float *ms, cudaEvent_t start, cudaEvent_t end)

Computes the elapsed time between events.

##### Parameters

**ms**

  - Time between start and end in ms
**start**

  - Starting event
**end**

  - Ending event

##### Returns

cudaSuccess, cudaErrorNotReady, cudaErrorInvalidValue, cudaErrorInvalidResourceHandle,
cudaErrorLaunchFailure, cudaErrorUnknown

##### Description

Computes the elapsed time between two events (in milliseconds with a resolution of around 0.5
microseconds). Note this API is not guaranteed to return the latest errors for pending work. As such
this API is intended to serve as a elapsed time calculation only and polling for completion on the events
to be compared should be done with cudaEventQuery instead.

If either event was last recorded in a non-NULL stream, the resulting time may be greater than
expected (even if both used the same stream handle). This happens because the cudaEventRecord()
operation takes place asynchronously and there is no guarantee that the measured latency is actually
just between the two events. Any number of other different stream operations could execute in between
the two measured events, thus altering the timing in a significant way.

If cudaEventRecord() has not been called on either event, then cudaErrorInvalidResourceHandle is
returned. If cudaEventRecord() has been called on both events but one or both of them has not yet been
completed (that is, cudaEventQuery() would return cudaErrorNotReady on at least one of the events),
cudaErrorNotReady is returned. If either event was created with the cudaEventDisableTiming flag,
then this function will return cudaErrorInvalidResourceHandle.



CUDA Runtime API vRelease Version  |  76


Modules


**‣** Returns cudaErrorInvalidResourceHandle in the event of being passed NULL as the input event.


See also:

cudaEventCreate ( C API), cudaEventCreateWithFlags, cudaEventQuery, cudaEventSynchronize,
cudaEventDestroy, cudaEventRecord, cuEventElapsedTime
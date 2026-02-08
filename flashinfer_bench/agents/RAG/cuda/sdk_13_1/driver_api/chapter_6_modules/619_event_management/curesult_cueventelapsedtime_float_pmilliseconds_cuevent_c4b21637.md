# CUresult cuEventElapsedTime (float *pMilliseconds, CUevent hStart, CUevent hEnd)

Computes the elapsed time between two events.

###### Parameters

**pMilliseconds**

  - Time between hStart and hEnd in ms
**hStart**

  - Starting event


CUDA Driver API TRM-06703-001 _vRelease Version  |  356


Modules


**hEnd**

  - Ending event

###### Returns

CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_HANDLE,
CUDA_ERROR_NOT_READY, CUDA_ERROR_UNKNOWN

###### Description

Computes the elapsed time between two events (in milliseconds with a resolution of around 0.5
microseconds). Note this API is not guaranteed to return the latest errors for pending work. As such
this API is intended to serve as an elapsed time calculation only and any polling for completion on the
events to be compared should be done with cuEventQuery instead.

If either event was last recorded in a non-NULL stream, the resulting time may be greater than
expected (even if both used the same stream handle). This happens because the cuEventRecord()
operation takes place asynchronously and there is no guarantee that the measured latency is actually
just between the two events. Any number of other different stream operations could execute in between
the two measured events, thus altering the timing in a significant way.

If cuEventRecord() has not been called on either event then CUDA_ERROR_INVALID_HANDLE
is returned. If cuEventRecord() has been called on both events but one or both of them has not
yet been completed (that is, cuEventQuery() would return CUDA_ERROR_NOT_READY
on at least one of the events), CUDA_ERROR_NOT_READY is returned. If either event
was created with the CU_EVENT_DISABLE_TIMING flag, then this function will return
CUDA_ERROR_INVALID_HANDLE.


Note:


Note that this function may also return error codes from previous, asynchronous launches.


See also:

cuEventCreate, cuEventRecord, cuEventQuery, cuEventSynchronize, cuEventDestroy,
cudaEventElapsedTime
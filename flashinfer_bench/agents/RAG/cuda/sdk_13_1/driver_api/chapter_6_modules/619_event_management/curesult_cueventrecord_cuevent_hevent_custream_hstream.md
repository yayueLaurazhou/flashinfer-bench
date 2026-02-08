# CUresult cuEventRecord (CUevent hEvent, CUstream hStream)

Records an event.

###### Parameters

**hEvent**

  - Event to record
**hStream**

  - Stream to record event for

###### Returns

CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_HANDLE,
CUDA_ERROR_INVALID_VALUE


CUDA Driver API TRM-06703-001 _vRelease Version  |  358


Modules

###### Description

Captures in hEvent the contents of hStream at the time of this call. hEvent and hStream must
be from the same context otherwise CUDA_ERROR_INVALID_HANDLE is returned. Calls such as
cuEventQuery() or cuStreamWaitEvent() will then examine or wait for completion of the work that was
captured. Uses of hStream after this call do not modify hEvent. See note on default stream behavior
for what is captured in the default case.

cuEventRecord() can be called multiple times on the same event and will overwrite the previously
captured state. Other APIs such as cuStreamWaitEvent() use the most recently captured state at the
time of the API call, and are not affected by later calls to cuEventRecord(). Before the first call to
cuEventRecord(), an event represents an empty set of work, so for example cuEventQuery() would
return CUDA_SUCCESS.





See also:

cuEventCreate, cuEventQuery, cuEventSynchronize, cuStreamWaitEvent, cuEventDestroy,
cuEventElapsedTime, cudaEventRecord, cuEventRecordWithFlags
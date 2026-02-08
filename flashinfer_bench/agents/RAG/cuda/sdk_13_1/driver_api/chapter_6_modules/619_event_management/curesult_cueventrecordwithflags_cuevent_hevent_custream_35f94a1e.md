# CUresult cuEventRecordWithFlags (CUevent hEvent, CUstream hStream, unsigned int flags)

Records an event.

###### Parameters

**hEvent**

  - Event to record
**hStream**

  - Stream to record event for
**flags**

  - See CUevent_capture_flags

###### Returns

CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_HANDLE,
CUDA_ERROR_INVALID_VALUE


CUDA Driver API TRM-06703-001 _vRelease Version  |  359


Modules

###### Description

Captures in hEvent the contents of hStream at the time of this call. hEvent and hStream must
be from the same context otherwise CUDA_ERROR_INVALID_HANDLE is returned. Calls such as
cuEventQuery() or cuStreamWaitEvent() will then examine or wait for completion of the work that was
captured. Uses of hStream after this call do not modify hEvent. See note on default stream behavior
for what is captured in the default case.

cuEventRecordWithFlags() can be called multiple times on the same event and will overwrite the
previously captured state. Other APIs such as cuStreamWaitEvent() use the most recently captured
state at the time of the API call, and are not affected by later calls to cuEventRecordWithFlags().
Before the first call to cuEventRecordWithFlags(), an event represents an empty set of work, so for
example cuEventQuery() would return CUDA_SUCCESS.

flags include:

CU_EVENT_RECORD_DEFAULT: Default event creation flag.

###### **‣**

CU_EVENT_RECORD_EXTERNAL: Event is captured in the graph as an external event node

###### **‣**

when performing stream capture. This flag is invalid outside of stream capture.





See also:

cuEventCreate, cuEventQuery, cuEventSynchronize, cuStreamWaitEvent, cuEventDestroy,
cuEventElapsedTime, cuEventRecord, cudaEventRecord
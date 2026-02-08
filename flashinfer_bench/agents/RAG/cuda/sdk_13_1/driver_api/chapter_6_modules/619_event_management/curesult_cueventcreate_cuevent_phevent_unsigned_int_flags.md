# CUresult cuEventCreate (CUevent *phEvent, unsigned int Flags)

Creates an event.

###### Parameters

**phEvent**

  - Returns newly created event
**Flags**

  - Event creation flags

###### Returns

CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE,
CUDA_ERROR_OUT_OF_MEMORY

###### Description

Creates an event *phEvent for the current context with the flags specified via Flags. Valid flags
include:

CU_EVENT_DEFAULT: Default event creation flag.

###### **‣**

CU_EVENT_BLOCKING_SYNC: Specifies that the created event should use blocking

###### **‣**

synchronization. A CPU thread that uses cuEventSynchronize() to wait on an event created with
this flag will block until the event has actually been recorded.
CU_EVENT_DISABLE_TIMING: Specifies that the created event does not need to record

###### **‣**

timing data. Events created with this flag specified and the CU_EVENT_BLOCKING_SYNC
flag not specified will provide the best performance when used with cuStreamWaitEvent() and
cuEventQuery().
CU_EVENT_INTERPROCESS: Specifies that the created event may be used as an interprocess

###### **‣**

event by cuIpcGetEventHandle(). CU_EVENT_INTERPROCESS must be specified along with
CU_EVENT_DISABLE_TIMING.


Note:


Note that this function may also return error codes from previous, asynchronous launches.


CUDA Driver API TRM-06703-001 _vRelease Version  |  355


Modules


See also:

cuEventRecord, cuEventQuery, cuEventSynchronize, cuEventDestroy, cuEventElapsedTime,
cudaEventCreate, cudaEventCreateWithFlags
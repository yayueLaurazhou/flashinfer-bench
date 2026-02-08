# __host____device__cudaError_t cudaEventCreateWithFlags (cudaEvent_t *event, unsigned int flags)

Creates an event object with the specified flags.

##### Parameters

**event**

  - Newly created event
**flags**

  - Flags for new event

##### Returns

cudaSuccess, cudaErrorInvalidValue, cudaErrorLaunchFailure, cudaErrorMemoryAllocation

##### Description

Creates an event object for the current device with the specified flags. Valid flags include:

cudaEventDefault: Default event creation flag.

##### **‣**

cudaEventBlockingSync: Specifies that event should use blocking synchronization. A host thread

##### **‣**

that uses cudaEventSynchronize() to wait on an event created with this flag will block until the
event actually completes.
cudaEventDisableTiming: Specifies that the created event does not need to record timing data.

##### **‣**

Events created with this flag specified and the cudaEventBlockingSync flag not specified will
provide the best performance when used with cudaStreamWaitEvent() and cudaEventQuery().
cudaEventInterprocess: Specifies that the created event may be used as an interprocess

##### **‣**

event by cudaIpcGetEventHandle(). cudaEventInterprocess must be specified along with
cudaEventDisableTiming.





See also:


CUDA Runtime API vRelease Version  |  74


Modules


cudaEventCreate ( C API), cudaEventSynchronize, cudaEventDestroy, cudaEventElapsedTime,
cudaStreamWaitEvent, cuEventCreate
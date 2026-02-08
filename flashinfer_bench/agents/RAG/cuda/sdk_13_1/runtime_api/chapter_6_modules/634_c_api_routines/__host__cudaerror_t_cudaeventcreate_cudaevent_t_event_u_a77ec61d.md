# __host__cudaError_t cudaEventCreate (cudaEvent_t *event, unsigned int flags)

[C++ API] Creates an event object with the specified flags

##### Parameters

**event**

  - Newly created event
**flags**

  - Flags for new event

##### Returns

cudaSuccess, cudaErrorInvalidValue, cudaErrorLaunchFailure, cudaErrorMemoryAllocation

##### Description

Creates an event object with the specified flags. Valid flags include:

cudaEventDefault: Default event creation flag.

##### **‣**

cudaEventBlockingSync: Specifies that event should use blocking synchronization. A host thread

##### **‣**

that uses cudaEventSynchronize() to wait on an event created with this flag will block until the
event actually completes.


CUDA Runtime API vRelease Version  |  463


Modules


cudaEventDisableTiming: Specifies that the created event does not need to record timing data.

##### **‣**

Events created with this flag specified and the cudaEventBlockingSync flag not specified will
provide the best performance when used with cudaStreamWaitEvent() and cudaEventQuery().



See also:

cudaEventCreate ( C API), cudaEventCreateWithFlags, cudaEventRecord, cudaEventQuery,
cudaEventSynchronize, cudaEventDestroy, cudaEventElapsedTime, cudaStreamWaitEvent
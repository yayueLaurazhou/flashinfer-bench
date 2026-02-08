# __host__cudaError_t cudaEventQuery (cudaEvent_t event)

Queries an event's status.

##### Parameters

**event**

  - Event to query

##### Returns

cudaSuccess, cudaErrorNotReady, cudaErrorInvalidValue, cudaErrorInvalidResourceHandle,
cudaErrorLaunchFailure

##### Description

Queries the status of all work currently captured by event. See cudaEventRecord() for details on what
is captured by an event.

Returns cudaSuccess if all captured work has been completed, or cudaErrorNotReady if any captured
work is incomplete.

For the purposes of Unified Memory, a return value of cudaSuccess is equivalent to having called
cudaEventSynchronize().









See also:

cudaEventCreate ( C API), cudaEventCreateWithFlags, cudaEventRecord, cudaEventSynchronize,
cudaEventDestroy, cudaEventElapsedTime, cuEventQuery


CUDA Runtime API vRelease Version  |  77


Modules
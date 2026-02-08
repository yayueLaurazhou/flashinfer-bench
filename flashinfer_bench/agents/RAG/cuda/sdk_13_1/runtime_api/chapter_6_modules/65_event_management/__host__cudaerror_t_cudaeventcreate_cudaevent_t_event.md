# __host__cudaError_t cudaEventCreate (cudaEvent_t *event)

Creates an event object.

##### Parameters

**event**

  - Newly created event

##### Returns

cudaSuccess, cudaErrorInvalidValue, cudaErrorLaunchFailure, cudaErrorMemoryAllocation

##### Description

Creates an event object for the current device using cudaEventDefault.





See also:

cudaEventCreate ( C++ API), cudaEventCreateWithFlags, cudaEventRecord, cudaEventQuery,
cudaEventSynchronize, cudaEventDestroy, cudaEventElapsedTime, cudaStreamWaitEvent,
cuEventCreate


CUDA Runtime API vRelease Version  |  73


Modules
# __host____device__cudaError_t cudaEventDestroy (cudaEvent_t event)

Destroys an event object.

##### Parameters

**event**

  - Event to destroy

##### Returns

cudaSuccess, cudaErrorInvalidValue, cudaErrorInvalidResourceHandle, cudaErrorLaunchFailure

##### Description

Destroys the event specified by event.

An event may be destroyed before it is complete (i.e., while cudaEventQuery() would return
cudaErrorNotReady). In this case, the call does not block on completion of the event, and any
associated resources will automatically be released asynchronously at completion.









See also:

cudaEventCreate ( C API), cudaEventCreateWithFlags, cudaEventQuery, cudaEventSynchronize,
cudaEventRecord, cudaEventElapsedTime, cuEventDestroy


CUDA Runtime API vRelease Version  |  75


Modules
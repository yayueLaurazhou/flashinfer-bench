# __host__cudaError_t cudaEventSynchronize (cudaEvent_t event)

Waits for an event to complete.

##### Parameters

**event**

  - Event to wait for

##### Returns

cudaSuccess, cudaErrorInvalidValue, cudaErrorInvalidResourceHandle, cudaErrorLaunchFailure

##### Description

Waits until the completion of all work currently captured in event. See cudaEventRecord() for details
on what is captured by an event.

Waiting for an event that was created with the cudaEventBlockingSync flag will cause the calling CPU
thread to block until the event has been completed by the device. If the cudaEventBlockingSync flag
has not been set, then the CPU thread will busy-wait until the event has been completed by the device.





See also:

cudaEventCreate ( C API), cudaEventCreateWithFlags, cudaEventRecord, cudaEventQuery,
cudaEventDestroy, cudaEventElapsedTime, cuEventSynchronize


CUDA Runtime API vRelease Version  |  80


Modules
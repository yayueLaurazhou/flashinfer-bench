# CUresult cuEventSynchronize (CUevent hEvent)

Waits for an event to complete.

###### Parameters

**hEvent**

  - Event to wait for

###### Returns

CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_HANDLE


CUDA Driver API TRM-06703-001 _vRelease Version  |  360


Modules

###### Description

Waits until the completion of all work currently captured in hEvent. See cuEventRecord() for details
on what is captured by an event.

Waiting for an event that was created with the CU_EVENT_BLOCKING_SYNC flag will
cause the calling CPU thread to block until the event has been completed by the device. If the
CU_EVENT_BLOCKING_SYNC flag has not been set, then the CPU thread will busy-wait until the
event has been completed by the device.


Note:


Note that this function may also return error codes from previous, asynchronous launches.


See also:

cuEventCreate, cuEventRecord, cuEventQuery, cuEventDestroy, cuEventElapsedTime,
cudaEventSynchronize
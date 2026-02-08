# CUresult cuStreamWaitEvent (CUstream hStream, CUevent hEvent, unsigned int Flags)

Make a compute stream wait on an event.

###### Parameters

**hStream**

  - Stream to wait
**hEvent**

  - Event to wait on (may not be NULL)
**Flags**

  - See CUevent_capture_flags


CUDA Driver API TRM-06703-001 _vRelease Version  |  352


Modules

###### Returns

CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_HANDLE,

###### Description

Makes all future work submitted to hStream wait for all work captured in hEvent. See
cuEventRecord() for details on what is captured by an event. The synchronization will be performed
efficiently on the device when applicable. hEvent may be from a different context or device than
hStream.

flags include:

CU_EVENT_WAIT_DEFAULT: Default event creation flag.

###### **‣**

CU_EVENT_WAIT_EXTERNAL: Event is captured in the graph as an external event node when

###### **‣**

performing stream capture. This flag is invalid outside of stream capture.





See also:

cuStreamCreate, cuEventRecord, cuStreamQuery, cuStreamSynchronize, cuStreamAddCallback,
cuStreamDestroy, cudaStreamWaitEvent
# __host__cudaError_t cudaEventCreateFromEGLSync (cudaEvent_t *phEvent, EGLSyncKHR eglSync, unsigned int flags)

Creates an event from EGLSync object.

##### Parameters

**phEvent**

  - Returns newly created event


CUDA Runtime API vRelease Version  |  292


Modules


**eglSync**

  - Opaque handle to EGLSync object
**flags**

  - Event creation flags

##### Returns

cudaSuccess, cudaErrorInitializationError, cudaErrorInvalidValue, cudaErrorLaunchFailure,
cudaErrorMemoryAllocation

##### Description

Creates an event *phEvent from an EGLSyncKHR eglSync with the flages specified via flags. Valid
flags include:

cudaEventDefault: Default event creation flag.

##### **‣**

cudaEventBlockingSync: Specifies that the created event should use blocking synchronization. A

##### **‣**

CPU thread that uses cudaEventSynchronize() to wait on an event created with this flag will block
until the event has actually been completed.

cudaEventRecord and TimingData are not supported for events created from EGLSync.

The EGLSyncKHR is an opaque handle to an EGL sync object. typedef void* EGLSyncKHR


See also:

cudaEventQuery, cudaEventSynchronize, cudaEventDestroy
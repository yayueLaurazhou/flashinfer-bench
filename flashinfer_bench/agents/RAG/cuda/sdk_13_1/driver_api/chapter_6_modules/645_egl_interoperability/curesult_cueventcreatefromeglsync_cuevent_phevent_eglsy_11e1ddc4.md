# CUresult cuEventCreateFromEGLSync (CUevent *phEvent, EGLSyncKHR eglSync, unsigned int flags)

Creates an event from EGLSync object.

###### Parameters

**phEvent**

  - Returns newly created event
**eglSync**

  - Opaque handle to EGLSync object
**flags**

  - Event creation flags


CUDA Driver API TRM-06703-001 _vRelease Version  |  666


Modules

###### Returns

CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE,
CUDA_ERROR_OUT_OF_MEMORY

###### Description

Creates an event *phEvent from an EGLSyncKHR eglSync with the flags specified via flags. Valid
flags include:

CU_EVENT_DEFAULT: Default event creation flag.

###### **‣**

CU_EVENT_BLOCKING_SYNC: Specifies that the created event should use blocking

###### **‣**

synchronization. A CPU thread that uses cuEventSynchronize() to wait on an event created with
this flag will block until the event has actually been completed.

Once the eglSync gets destroyed, cuEventDestroy is the only API that can be invoked on the event.

cuEventRecord and TimingData are not supported for events created from EGLSync.

The EGLSyncKHR is an opaque handle to an EGL sync object. typedef void* EGLSyncKHR


See also:

cuEventQuery, cuEventSynchronize, cuEventDestroy
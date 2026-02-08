# CUresult cuStreamIsCapturing (CUstream hStream, CUstreamCaptureStatus *captureStatus)

Returns a stream's capture status.

###### Parameters

**hStream**

  - Stream to query
**captureStatus**

  - Returns the stream's capture status


CUDA Driver API TRM-06703-001 _vRelease Version  |  348


Modules

###### Returns

CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
CUDA_ERROR_INVALID_VALUE, CUDA_ERROR_STREAM_CAPTURE_IMPLICIT

###### Description

Return the capture status of hStream via captureStatus. After a successful call,
*captureStatus will contain one of the following:

CU_STREAM_CAPTURE_STATUS_NONE: The stream is not capturing.

###### **‣**

CU_STREAM_CAPTURE_STATUS_ACTIVE: The stream is capturing.

###### **‣**

CU_STREAM_CAPTURE_STATUS_INVALIDATED: The stream was capturing but an

###### **‣**

error has invalidated the capture sequence. The capture sequence must be terminated with
cuStreamEndCapture on the stream where it was initiated in order to continue using hStream.

Note that, if this is called on CU_STREAM_LEGACY (the "null stream") while a blocking stream in
the same context is capturing, it will return CUDA_ERROR_STREAM_CAPTURE_IMPLICIT and
*captureStatus is unspecified after the call. The blocking stream capture is not invalidated.

When a blocking stream is capturing, the legacy stream is in an unusable state until the blocking stream
capture is terminated. The legacy stream is not supported for stream capture, but attempted use would
have an implicit dependency on the capturing stream(s).


Note:


Note that this function may also return error codes from previous, asynchronous launches.


See also:

cuStreamCreate, cuStreamBeginCapture, cuStreamEndCapture
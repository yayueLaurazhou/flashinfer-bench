# __host__cudaError_t cudaStreamIsCapturing (cudaStream_t stream, cudaStreamCaptureStatus *pCaptureStatus)

Returns a stream's capture status.

##### Parameters

**stream**

  - Stream to query
**pCaptureStatus**

  - Returns the stream's capture status

##### Returns

cudaSuccess, cudaErrorInvalidValue, cudaErrorStreamCaptureImplicit

##### Description

Return the capture status of stream via pCaptureStatus. After a successful call,
*pCaptureStatus will contain one of the following:

cudaStreamCaptureStatusNone: The stream is not capturing.

##### **‣**

cudaStreamCaptureStatusActive: The stream is capturing.

##### **‣**

cudaStreamCaptureStatusInvalidated: The stream was capturing but an error has invalidated the

##### **‣**

capture sequence. The capture sequence must be terminated with cudaStreamEndCapture on the
stream where it was initiated in order to continue using stream.

Note that, if this is called on cudaStreamLegacy (the "null stream") while a blocking stream on the
same device is capturing, it will return cudaErrorStreamCaptureImplicit and *pCaptureStatus is
unspecified after the call. The blocking stream capture is not invalidated.


CUDA Runtime API vRelease Version  |  66


Modules


When a blocking stream is capturing, the legacy stream is in an unusable state until the blocking stream
capture is terminated. The legacy stream is not supported for stream capture, but attempted use would
have an implicit dependency on the capturing stream(s).


Note:


Note that this function may also return error codes from previous, asynchronous launches.


See also:

cudaStreamCreate, cudaStreamBeginCapture, cudaStreamEndCapture
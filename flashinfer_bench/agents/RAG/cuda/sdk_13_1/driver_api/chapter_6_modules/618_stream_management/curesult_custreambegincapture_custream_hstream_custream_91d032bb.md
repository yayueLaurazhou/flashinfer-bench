# CUresult cuStreamBeginCapture (CUstream hStream, CUstreamCaptureMode mode)

Begins graph capture on a stream.

###### Parameters

**hStream**

  - Stream in which to initiate capture
**mode**

  - Controls the interaction of this capture sequence with other API calls that are potentially unsafe.
For more details see cuThreadExchangeStreamCaptureMode.

###### Returns

CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
CUDA_ERROR_INVALID_VALUE


CUDA Driver API TRM-06703-001 _vRelease Version  |  334


Modules

###### Description

Begin graph capture on hStream. When a stream is in capture mode, all operations pushed into
the stream will not be executed, but will instead be captured into a graph, which will be returned via
cuStreamEndCapture. Capture may not be initiated if stream is CU_STREAM_LEGACY. Capture
must be ended on the same stream in which it was initiated, and it may only be initiated if the stream is
not already in capture mode. The capture mode may be queried via cuStreamIsCapturing. A unique id
representing the capture sequence may be queried via cuStreamGetCaptureInfo.

If mode is not CU_STREAM_CAPTURE_MODE_RELAXED, cuStreamEndCapture must be called
on this stream from the same thread.


Note:


Kernels captured using this API must not use texture and surface references. Reading or writing through
any texture or surface reference is undefined behavior. This restriction does not apply to texture and
surface objects.


Note:


Note that this function may also return error codes from previous, asynchronous launches.


See also:

cuStreamCreate, cuStreamIsCapturing, cuStreamEndCapture, cuThreadExchangeStreamCaptureMode
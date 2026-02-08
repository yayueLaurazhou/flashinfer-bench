# __host__cudaError_t cudaStreamBeginCapture (cudaStream_t stream, cudaStreamCaptureMode mode)

Begins graph capture on a stream.

##### Parameters

**stream**

  - Stream in which to initiate capture
**mode**

  - Controls the interaction of this capture sequence with other API calls that are potentially unsafe.
For more details see cudaThreadExchangeStreamCaptureMode.

##### Returns

cudaSuccess, cudaErrorInvalidValue

##### Description

Begin graph capture on stream. When a stream is in capture mode, all operations pushed into the
stream will not be executed, but will instead be captured into a graph, which will be returned via
cudaStreamEndCapture. Capture may not be initiated if stream is cudaStreamLegacy. Capture must
be ended on the same stream in which it was initiated, and it may only be initiated if the stream is not
already in capture mode. The capture mode may be queried via cudaStreamIsCapturing. A unique id
representing the capture sequence may be queried via cudaStreamGetCaptureInfo.

If mode is not cudaStreamCaptureModeRelaxed, cudaStreamEndCapture must be called on this stream
from the same thread.


Note:


Kernels captured using this API must not use texture and surface references. Reading or writing through
any texture or surface reference is undefined behavior. This restriction does not apply to texture and
surface objects.


CUDA Runtime API vRelease Version  |  53


Modules


Note:


Note that this function may also return error codes from previous, asynchronous launches.


See also:

cudaStreamCreate, cudaStreamIsCapturing, cudaStreamEndCapture,
cudaThreadExchangeStreamCaptureMode
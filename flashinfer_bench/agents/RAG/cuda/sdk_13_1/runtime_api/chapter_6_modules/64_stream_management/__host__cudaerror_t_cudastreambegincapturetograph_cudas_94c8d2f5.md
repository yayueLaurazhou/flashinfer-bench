# __host__cudaError_t cudaStreamBeginCaptureToGraph (cudaStream_t stream, cudaGraph_t graph, const cudaGraphNode_t *dependencies, const cudaGraphEdgeData *dependencyData, size_t numDependencies, cudaStreamCaptureMode mode)

Begins graph capture on a stream to an existing graph.

##### Parameters

**stream**

  - Stream in which to initiate capture.
**graph**

  - Graph to capture into.
**dependencies**

  - Dependencies of the first node captured in the stream. Can be NULL if numDependencies is 0.
**dependencyData**

  - Optional array of data associated with each dependency.
**numDependencies**

  - Number of dependencies.
**mode**

  - Controls the interaction of this capture sequence with other API calls that are potentially unsafe.
For more details see cudaThreadExchangeStreamCaptureMode.

##### Returns

cudaSuccess, cudaErrorInvalidValue

##### Description

Begin graph capture on stream. When a stream is in capture mode, all operations pushed into the
stream will not be executed, but will instead be captured into graph, which will be returned via
cudaStreamEndCapture.


CUDA Runtime API vRelease Version  |  54


Modules


Capture may not be initiated if stream is cudaStreamLegacy. Capture must be ended on the same
stream in which it was initiated, and it may only be initiated if the stream is not already in capture
mode. The capture mode may be queried via cudaStreamIsCapturing. A unique id representing the
capture sequence may be queried via cudaStreamGetCaptureInfo.

If mode is not cudaStreamCaptureModeRelaxed, cudaStreamEndCapture must be called on this stream
from the same thread.


Note:


Kernels captured using this API must not use texture and surface references. Reading or writing through
any texture or surface reference is undefined behavior. This restriction does not apply to texture and
surface objects.


Note:


Note that this function may also return error codes from previous, asynchronous launches.


See also:

cudaStreamCreate, cudaStreamIsCapturing, cudaStreamEndCapture,
cudaThreadExchangeStreamCaptureMode
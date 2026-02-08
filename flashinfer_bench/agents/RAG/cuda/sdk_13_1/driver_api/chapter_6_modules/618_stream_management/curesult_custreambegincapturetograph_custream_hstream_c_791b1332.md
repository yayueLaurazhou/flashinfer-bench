# CUresult cuStreamBeginCaptureToGraph (CUstream hStream, CUgraph hGraph, const CUgraphNode *dependencies, const CUgraphEdgeData *dependencyData, size_t numDependencies, CUstreamCaptureMode mode)

Begins graph capture on a stream to an existing graph.

###### Parameters

**hStream**

  - Stream in which to initiate capture.
**hGraph**

  - Graph to capture into.
**dependencies**

  - Dependencies of the first node captured in the stream. Can be NULL if numDependencies is 0.


CUDA Driver API TRM-06703-001 _vRelease Version  |  335


Modules


**dependencyData**

  - Optional array of data associated with each dependency.
**numDependencies**

  - Number of dependencies.
**mode**

  - Controls the interaction of this capture sequence with other API calls that are potentially unsafe.
For more details see cuThreadExchangeStreamCaptureMode.

###### Returns

CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
CUDA_ERROR_INVALID_VALUE

###### Description

Begin graph capture on hStream, placing new nodes into an existing graph. When a stream is in
capture mode, all operations pushed into the stream will not be executed, but will instead be captured
into hGraph. The graph will not be instantiable until the user calls cuStreamEndCapture.

Capture may not be initiated if stream is CU_STREAM_LEGACY. Capture must be ended on
the same stream in which it was initiated, and it may only be initiated if the stream is not already in
capture mode. The capture mode may be queried via cuStreamIsCapturing. A unique id representing
the capture sequence may be queried via cuStreamGetCaptureInfo.

If mode is not CU_STREAM_CAPTURE_MODE_RELAXED, cuStreamEndCapture must be called
on this stream from the same thread.


Note:


Kernels captured using this API must not use texture and surface references. Reading or writing through
any texture or surface reference is undefined behavior. This restriction does not apply to texture and
surface objects.


Note:


Note that this function may also return error codes from previous, asynchronous launches.


See also:

cuStreamBeginCapture, cuStreamCreate, cuStreamIsCapturing, cuStreamEndCapture,
cuThreadExchangeStreamCaptureMode, cuGraphAddNode


CUDA Driver API TRM-06703-001 _vRelease Version  |  336


Modules
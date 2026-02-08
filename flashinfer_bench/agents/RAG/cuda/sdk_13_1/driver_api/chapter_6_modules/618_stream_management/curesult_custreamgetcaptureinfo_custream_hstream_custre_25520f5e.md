# CUresult cuStreamGetCaptureInfo (CUstream hStream, CUstreamCaptureStatus *captureStatus_out, cuuint64_t *id_out, CUgraph *graph_out, const CUgraphNode **dependencies_out, const CUgraphEdgeData **edgeData_out, size_t *numDependencies_out)

Query a stream's capture state.

###### Parameters

**hStream**

  - The stream to query
**captureStatus_out**

  - Location to return the capture status of the stream; required


CUDA Driver API TRM-06703-001 _vRelease Version  |  341


Modules


**id_out**

  - Optional location to return an id for the capture sequence, which is unique over the lifetime of the
process
**graph_out**

  - Optional location to return the graph being captured into. All operations other than destroy and
node removal are permitted on the graph while the capture sequence is in progress. This API does
not transfer ownership of the graph, which is transferred or destroyed at cuStreamEndCapture. Note
that the graph handle may be invalidated before end of capture for certain errors. Nodes that are or
become unreachable from the original stream at cuStreamEndCapture due to direct actions on the
graph do not trigger CUDA_ERROR_STREAM_CAPTURE_UNJOINED.
**dependencies_out**

  - Optional location to store a pointer to an array of nodes. The next node to be captured in the
stream will depend on this set of nodes, absent operations such as event wait which modify this set.
The array pointer is valid until the next API call which operates on the stream or until the capture is
terminated. The node handles may be copied out and are valid until they or the graph is destroyed.
The driver-owned array may also be passed directly to APIs that operate on the graph (not the
stream) without copying.
**edgeData_out**

  - Optional location to store a pointer to an array of graph edge data. This array parallels
dependencies_out; the next node to be added has an edge to dependencies_out[i] with
annotation edgeData_out[i] for each i. The array pointer is valid until the next API call which
operates on the stream or until the capture is terminated.
**numDependencies_out**

  - Optional location to store the size of the array returned in dependencies_out.

###### Returns

CUDA_SUCCESS, CUDA_ERROR_INVALID_VALUE,
CUDA_ERROR_STREAM_CAPTURE_IMPLICIT, CUDA_ERROR_LOSSY_QUERY

###### Description

Query stream state related to stream capture.

If called on CU_STREAM_LEGACY (the "null stream") while a stream
not created with CU_STREAM_NON_BLOCKING is capturing, returns
CUDA_ERROR_STREAM_CAPTURE_IMPLICIT.

Valid data (other than capture status) is returned only if both of the following are true:

the call returns CUDA_SUCCESS

###### **‣**

the returned capture status is CU_STREAM_CAPTURE_STATUS_ACTIVE

###### **‣**

If edgeData_out is non-NULL then dependencies_out must be as well. If
dependencies_out is non-NULL and edgeData_out is NULL, but there is nonzero edge data for one or more of the current stream dependencies, the call will return
CUDA_ERROR_LOSSY_QUERY.


CUDA Driver API TRM-06703-001 _vRelease Version  |  342


Modules





See also:

cuStreamBeginCapture, cuStreamIsCapturing, cuStreamUpdateCaptureDependencies
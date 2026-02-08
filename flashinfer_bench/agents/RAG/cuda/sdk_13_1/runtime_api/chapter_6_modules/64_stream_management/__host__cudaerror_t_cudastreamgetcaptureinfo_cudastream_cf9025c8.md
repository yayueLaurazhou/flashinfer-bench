# __host__cudaError_t cudaStreamGetCaptureInfo (cudaStream_t stream, cudaStreamCaptureStatus *captureStatus_out, unsigned long long *id_out, cudaGraph_t *graph_out, const cudaGraphNode_t **dependencies_out, const cudaGraphEdgeData **edgeData_out, size_t *numDependencies_out)

Query a stream's capture state.

##### Parameters

**stream**

  - The stream to query
**captureStatus_out**

  - Location to return the capture status of the stream; required
**id_out**

  - Optional location to return an id for the capture sequence, which is unique over the lifetime of the
process
**graph_out**

  - Optional location to return the graph being captured into. All operations other than destroy and
node removal are permitted on the graph while the capture sequence is in progress. This API does
not transfer ownership of the graph, which is transferred or destroyed at cudaStreamEndCapture.
Note that the graph handle may be invalidated before end of capture for certain errors. Nodes that
are or become unreachable from the original stream at cudaStreamEndCapture due to direct actions
on the graph do not trigger cudaErrorStreamCaptureUnjoined.


CUDA Runtime API vRelease Version  |  61


Modules


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

##### Returns

cudaSuccess, cudaErrorInvalidValue, cudaErrorStreamCaptureImplicit, cudaErrorLossyQuery

##### Description

Query stream state related to stream capture.

If called on cudaStreamLegacy (the "null stream") while a stream not created with
cudaStreamNonBlocking is capturing, returns cudaErrorStreamCaptureImplicit.

Valid data (other than capture status) is returned only if both of the following are true:

the call returns cudaSuccess

##### **‣**

the returned capture status is cudaStreamCaptureStatusActive

##### **‣**

If edgeData_out is non-NULL then dependencies_out must be as well. If
dependencies_out is non-NULL and edgeData_out is NULL, but there is non-zero edge data
for one or more of the current stream dependencies, the call will return cudaErrorLossyQuery.





See also:

cudaStreamBeginCapture, cudaStreamIsCapturing, cudaStreamUpdateCaptureDependencies


CUDA Runtime API vRelease Version  |  62


Modules
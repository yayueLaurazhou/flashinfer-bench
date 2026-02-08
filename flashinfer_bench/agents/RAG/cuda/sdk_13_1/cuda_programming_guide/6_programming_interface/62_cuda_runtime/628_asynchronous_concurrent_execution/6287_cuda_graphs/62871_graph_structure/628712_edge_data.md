# 6.2.8.7.1.2. Edge Data

###### 6.2.8.7.1.2. Edge Data[](#edge-data "Permalink to this headline")

CUDA 12.3 introduced edge data on CUDA Graphs. Edge data modifies a dependency specified by an edge and consists of three parts:
an outgoing port, an incoming port, and a type. An outgoing port specifies when an associated edge is triggered. An incoming port
specifies what portion of a node is dependent on an associated edge. A type modifies the relation between the endpoints.

Port values are specific to node type and direction, and edge types may be restricted to specific node types. In all cases,
zero-initialized edge data represents default behavior. Outgoing port 0 waits on an entire task, incoming port 0 blocks an
entire task, and edge type 0 is associated with a full dependency with memory synchronizing behavior.

Edge data is optionally specified in various graph APIs via a parallel array to the associated nodes. If it is omitted as an
input parameter, zero-initialized data is used. If it is omitted as an output (query) parameter, the API accepts this if the
edge data being ignored is all zero-initialized, and returns `cudaErrorLossyQuery` if the call would discard information.

Edge data is also available in some stream capture APIs: `cudaStreamBeginCaptureToGraph()`, `cudaStreamGetCaptureInfo()`,
and `cudaStreamUpdateCaptureDependencies()`. In these cases, there is not yet a downstream node. The data is associated with
a dangling edge (half edge) which will either be connected to a future captured node or discarded at termination of stream capture.
Note that some edge types do not wait on full completion of the upstream node. These edges are ignored when considering if a
stream capture has been fully rejoined to the origin stream, and cannot be discarded at the end of capture. See [Creating a Graph Using Stream Capture](#creating-a-graph-using-stream-capture).

Currently, no node types define additional incoming ports, and only kernel nodes define additional outgoing ports. There is
one non-default dependency type, `cudaGraphDependencyTypeProgrammatic`, which enables [Programmatic Dependent Launch](#programmatic-dependent-launch-and-synchronization) between two kernel nodes.
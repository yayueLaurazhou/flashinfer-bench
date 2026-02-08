# CUresult cuGraphInstantiateWithParams (CUgraphExec *phGraphExec, CUgraph hGraph, CUDA_GRAPH_INSTANTIATE_PARAMS *instantiateParams)

Creates an executable graph from a graph.

###### Parameters

**phGraphExec**

  - Returns instantiated graph
**hGraph**

  - Graph to instantiate
**instantiateParams**

  - Instantiation parameters

###### Returns

CUDA_SUCCESS, CUDA_ERROR_INVALID_VALUE,

###### Description

Instantiates hGraph as an executable graph according to the instantiateParams structure.
The graph is validated for any structural constraints or intra-node constraints which were not


CUDA Driver API TRM-06703-001 _vRelease Version  |  472


Modules


previously validated. If instantiation is successful, a handle to the instantiated graph is returned in
phGraphExec.

instantiateParams controls the behavior of instantiation and subsequent graph
launches, as well as returning more detailed information in the event of an error.
CUDA_GRAPH_INSTANTIATE_PARAMS is defined as:


The flags field controls the behavior of instantiation and subsequent graph launches. Valid flags are:

CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH, which configures a

###### **‣**

graph containing memory allocation nodes to automatically free any unfreed memory allocations
before the graph is relaunched.

CUDA_GRAPH_INSTANTIATE_FLAG_UPLOAD, which will perform an upload of the graph

###### **‣**

into hUploadStream once the graph has been instantiated.

CUDA_GRAPH_INSTANTIATE_FLAG_DEVICE_LAUNCH, which configures the graph

###### **‣**

for launch from the device. If this flag is passed, the executable graph handle returned can
be used to launch the graph from both the host and device. This flag can only be used on
platforms which support unified addressing. This flag cannot be used in conjunction with
CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH.

CUDA_GRAPH_INSTANTIATE_FLAG_USE_NODE_PRIORITY, which causes the graph to

###### **‣**

use the priorities from the per-node attributes rather than the priority of the launch stream during
execution. Note that priorities are only available on kernel nodes, and are copied from stream
priority during stream capture.

If hGraph contains any allocation or free nodes, there can be at most one executable graph in
existence for that graph at a time. An attempt to instantiate a second executable graph before destroying
the first with cuGraphExecDestroy will result in an error. The same also applies if hGraph contains
any device-updatable kernel nodes.

If hGraph contains kernels which call device-side cudaGraphLaunch() from multiple contexts, this
will result in an error.

Graphs instantiated for launch on the device have additional restrictions which do not apply to host
graphs:

The graph's nodes must reside on a single context.

###### **‣**

The graph can only contain kernel nodes, memcpy nodes, memset nodes, and child graph nodes.

###### **‣**

The graph cannot be empty and must contain at least one kernel, memcpy, or memset node.

###### **‣**

Operation-specific restrictions are outlined below.
Kernel nodes:

###### **‣**

Use of CUDA Dynamic Parallelism is not permitted.

###### **‣**

CUDA Driver API TRM-06703-001 _vRelease Version  |  473


Modules


Cooperative launches are permitted as long as MPS is not in use.

###### **‣**

Memcpy nodes:

###### **‣**

Only copies involving device memory and/or pinned device-mapped host memory are

###### **‣**

permitted.
Copies involving CUDA arrays are not permitted.

###### **‣**

Both operands must be accessible from the current context, and the current context must match

###### **‣**

the context of other nodes in the graph.

In the event of an error, the result_out and hErrNode_out fields will contain more information
about the nature of the error. Possible error reporting includes:

CUDA_GRAPH_INSTANTIATE_ERROR, if passed an invalid value or if an unexpected error

###### **‣**

occurred which is described by the return value of the function. hErrNode_out will be set to
NULL.
CUDA_GRAPH_INSTANTIATE_INVALID_STRUCTURE, if the graph structure is invalid.

###### **‣**

hErrNode_out will be set to one of the offending nodes.
CUDA_GRAPH_INSTANTIATE_NODE_OPERATION_NOT_SUPPORTED, if the graph is

###### **‣**

instantiated for device launch but contains a node of an unsupported node type, or a node which
performs unsupported operations, such as use of CUDA dynamic parallelism within a kernel node.
hErrNode_out will be set to this node.
CUDA_GRAPH_INSTANTIATE_MULTIPLE_CTXS_NOT_SUPPORTED, if the graph is

###### **‣**

instantiated for device launch but a node’s context differs from that of another node. This error can
also be returned if a graph is not instantiated for device launch and it contains kernels which call
device-side cudaGraphLaunch() from multiple contexts. hErrNode_out will be set to this node.

If instantiation is successful, result_out will be set to
CUDA_GRAPH_INSTANTIATE_SUCCESS, and hErrNode_out will be set to NULL.





See also:

cuGraphCreate, cuGraphInstantiate, cuGraphExecDestroy


CUDA Driver API TRM-06703-001 _vRelease Version  |  474


Modules
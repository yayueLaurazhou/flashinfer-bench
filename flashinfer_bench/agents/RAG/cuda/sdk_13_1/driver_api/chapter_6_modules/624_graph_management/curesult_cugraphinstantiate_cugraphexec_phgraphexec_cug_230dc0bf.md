# CUresult cuGraphInstantiate (CUgraphExec *phGraphExec, CUgraph hGraph, unsigned long long flags)

Creates an executable graph from a graph.

###### Parameters

**phGraphExec**

  - Returns instantiated graph
**hGraph**

  - Graph to instantiate


CUDA Driver API TRM-06703-001 _vRelease Version  |  470


Modules


**flags**

  - Flags to control instantiation. See CUgraphInstantiate_flags.

###### Returns

CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
CUDA_ERROR_INVALID_VALUE

###### Description

Instantiates hGraph as an executable graph. The graph is validated for any structural constraints or
intra-node constraints which were not previously validated. If instantiation is successful, a handle to the
instantiated graph is returned in phGraphExec.

The flags parameter controls the behavior of instantiation and subsequent graph launches. Valid
flags are:

CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH, which configures a

###### **‣**

graph containing memory allocation nodes to automatically free any unfreed memory allocations
before the graph is relaunched.

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

CUDA Driver API TRM-06703-001 _vRelease Version  |  471


Modules


Use of CUDA Dynamic Parallelism is not permitted.

###### **‣**

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





See also:

cuGraphInstantiate, cuGraphCreate, cuGraphUpload, cuGraphLaunch, cuGraphExecDestroy
# CUresult cuGraphExecChildGraphNodeSetParams (CUgraphExec hGraphExec, CUgraphNode hNode, CUgraph childGraph)

Updates node parameters in the child graph node in the given graphExec.

###### Parameters

**hGraphExec**

  - The executable graph in which to set the specified node
**hNode**

  - Host node from the graph which was used to instantiate graphExec
**childGraph**

  - The graph supplying the updated parameters

###### Returns

CUDA_SUCCESS, CUDA_ERROR_INVALID_VALUE,

###### Description

Updates the work represented by hNode in hGraphExec as though the nodes contained in hNode's
graph had the parameters contained in childGraph's nodes at instantiation. hNode must remain
in the graph which was used to instantiate hGraphExec. Changed edges to and from hNode are
ignored.

The modifications only affect future launches of hGraphExec. Already enqueued or running
launches of hGraphExec are not affected by this call. hNode is also not modified by this call.

The topology of childGraph, as well as the node insertion order, must match that of the graph
contained in hNode. See cuGraphExecUpdate() for a list of restrictions on what can be updated in an
instantiated graph. The update is recursive, so child graph nodes contained within the top level child
graph will also be updated.


CUDA Driver API TRM-06703-001 _vRelease Version  |  446


Modules





See also:

cuGraphExecNodeSetParams, cuGraphAddChildGraphNode,
cuGraphChildGraphNodeGetGraph, cuGraphExecKernelNodeSetParams,
cuGraphExecMemcpyNodeSetParams, cuGraphExecMemsetNodeSetParams,
cuGraphExecHostNodeSetParams, cuGraphExecEventRecordNodeSetEvent,
cuGraphExecEventWaitNodeSetEvent, cuGraphExecExternalSemaphoresSignalNodeSetParams,
cuGraphExecExternalSemaphoresWaitNodeSetParams, cuGraphExecUpdate, cuGraphInstantiate
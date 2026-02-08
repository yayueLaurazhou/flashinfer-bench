# __host__cudaError_t cudaGraphExecChildGraphNodeSetParams (cudaGraphExec_t hGraphExec, cudaGraphNode_t node, cudaGraph_t childGraph)

Updates node parameters in the child graph node in the given graphExec.

##### Parameters

**hGraphExec**

  - The executable graph in which to set the specified node
**node**

  - Host node from the graph which was used to instantiate graphExec
**childGraph**

  - The graph supplying the updated parameters

##### Returns

cudaSuccess, cudaErrorInvalidValue,

##### Description

Updates the work represented by node in hGraphExec as though the nodes contained in node's
graph had the parameters contained in childGraph's nodes at instantiation. node must remain in
the graph which was used to instantiate hGraphExec. Changed edges to and from node are ignored.

The modifications only affect future launches of hGraphExec. Already enqueued or running
launches of hGraphExec are not affected by this call. node is also not modified by this call.

The topology of childGraph, as well as the node insertion order, must match that of the graph
contained in node. See cudaGraphExecUpdate() for a list of restrictions on what can be updated in an
instantiated graph. The update is recursive, so child graph nodes contained within the top level child
graph will also be updated.


CUDA Runtime API vRelease Version  |  357


Modules











See also:

cudaGraphExecNodeSetParams, cudaGraphAddChildGraphNode,
cudaGraphChildGraphNodeGetGraph, cudaGraphExecKernelNodeSetParams,
cudaGraphExecMemcpyNodeSetParams, cudaGraphExecMemsetNodeSetParams,
cudaGraphExecHostNodeSetParams, cudaGraphExecEventRecordNodeSetEvent,
cudaGraphExecEventWaitNodeSetEvent, cudaGraphExecExternalSemaphoresSignalNodeSetParams,
cudaGraphExecExternalSemaphoresWaitNodeSetParams, cudaGraphExecUpdate,
cudaGraphInstantiate
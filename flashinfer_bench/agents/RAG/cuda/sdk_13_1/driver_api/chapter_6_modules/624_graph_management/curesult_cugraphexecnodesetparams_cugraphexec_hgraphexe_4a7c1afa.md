# CUresult cuGraphExecNodeSetParams (CUgraphExec hGraphExec, CUgraphNode hNode, CUgraphNodeParams *nodeParams)

Update's a graph node's parameters in an instantiated graph.

###### Parameters

**hGraphExec**

  - The executable graph in which to update the specified node
**hNode**

  - Corresponding node from the graph from which graphExec was instantiated
**nodeParams**

  - Updated Parameters to set

###### Returns

CUDA_SUCCESS, CUDA_ERROR_INVALID_VALUE, CUDA_ERROR_NOT_SUPPORTED

###### Description

Sets the parameters of a node in an executable graph hGraphExec. The node is identified by
the corresponding node hNode in the non-executable graph from which the executable graph was
instantiated. hNode must not have been removed from the original graph.

The modifications only affect future launches of hGraphExec. Already enqueued or running
launches of hGraphExec are not affected by this call. hNode is also not modified by this call.

Allowed changes to parameters on executable graphs are as follows:


CUDA Driver API TRM-06703-001 _vRelease Version  |  458


Modules





See also:

cuGraphAddNode, cuGraphNodeSetParams cuGraphExecUpdate, cuGraphInstantiate


CUDA Driver API TRM-06703-001 _vRelease Version  |  459


Modules
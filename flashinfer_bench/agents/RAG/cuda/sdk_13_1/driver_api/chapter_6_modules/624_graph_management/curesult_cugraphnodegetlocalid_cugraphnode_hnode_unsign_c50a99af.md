# CUresult cuGraphNodeGetLocalId (CUgraphNode hNode, unsigned int *nodeId)

Returns the local node id of a given graph node.

###### Parameters

**hNode**

  - Node to query
**nodeId**

  - Pointer to return the nodeId

###### Returns

CUDA_SUCCESS CUDA_ERROR_INVALID_VALUE

###### Description

Returns the node id of hNode in *nodeId. The nodeId matches that referenced by
cuGraphDebugDotPrint. The local nodeId and graphId together can uniquely identify the node.


See also:

cuGraphGetNodes, cuGraphDebugDotPrint cuGraphNodeGetContainingGraph
cuGraphNodeGetToolsId cuGraphGetId cuGraphExecGetId
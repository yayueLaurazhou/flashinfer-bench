# __host__cudaError_t cudaGraphNodeGetLocalId (cudaGraphNode_t hNode, unsigned int *nodeId)

Returns the node id of a given graph node.

##### Parameters

**hNode**

  - Node to query
**nodeId**

  - Pointer to return the nodeId


CUDA Runtime API vRelease Version  |  418


Modules

##### Returns

cudaSuccess cudaErrorInvalidValue

##### Description

Returns the node id of hNode in *nodeId. The nodeId matches that referenced by
cudaGraphDebugDotPrint. The local nodeId and graphId together can uniquely identify the node.


See also:

cudaGraphGetNodes, cudaGraphDebugDotPrint cudaGraphNodeGetContainingGraph
cudaGraphNodeGetToolsId cudaGraphGetId cudaGraphExecGetId
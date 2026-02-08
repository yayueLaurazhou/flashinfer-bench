# __host__cudaError_t cudaGraphNodeGetContainingGraph (cudaGraphNode_t hNode, cudaGraph_t *phGraph)

Returns the graph that contains a given graph node.

##### Parameters

**hNode**

  - Node to query
**phGraph**

  - Pointer to return the containing graph

##### Returns

cudaSuccess cudaErrorInvalidValue

##### Description

Returns the graph that contains hNode in *phGraph. If hNode is in a child graph, the child graph it is
in is returned.


See also:

cudaGraphGetNodes, cudaGraphDebugDotPrint cudaGraphNodeGetLocalId
cudaGraphNodeGetToolsId cudaGraphGetId cudaGraphExecGetId
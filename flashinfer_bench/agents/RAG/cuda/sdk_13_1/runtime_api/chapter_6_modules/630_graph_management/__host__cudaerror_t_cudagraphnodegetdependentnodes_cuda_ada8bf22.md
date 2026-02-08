# __host__cudaError_t cudaGraphNodeGetDependentNodes (cudaGraphNode_t node, cudaGraphNode_t *pDependentNodes, cudaGraphEdgeData *edgeData, size_t *pNumDependentNodes)

Returns a node's dependent nodes.

##### Parameters

**node**

  - Node to query
**pDependentNodes**

  - Pointer to return the dependent nodes


CUDA Runtime API vRelease Version  |  416


Modules


**edgeData**

  - Optional pointer to return edge data for dependent nodes
**pNumDependentNodes**

  - See description

##### Returns

cudaSuccess, cudaErrorLossyQuery, cudaErrorInvalidValue

##### Description

Returns a list of node's dependent nodes. pDependentNodes may be NULL, in which case
this function will return the number of dependent nodes in pNumDependentNodes. Otherwise,
pNumDependentNodes entries will be filled in. If pNumDependentNodes is higher than the
actual number of dependent nodes, the remaining entries in pDependentNodes will be set to NULL,
and the number of nodes actually obtained will be returned in pNumDependentNodes.

Note that if an edge has non-zero (non-default) edge data and edgeData is NULL, this API will
return cudaErrorLossyQuery. If edgeData is non-NULL, then pDependentNodes must be as
well.











See also:

cudaGraphNodeGetDependencies, cudaGraphGetNodes, cudaGraphGetRootNodes,
cudaGraphGetEdges, cudaGraphAddDependencies, cudaGraphRemoveDependencies
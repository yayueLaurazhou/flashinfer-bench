# __host__cudaError_t cudaGraphGetRootNodes (cudaGraph_t graph, cudaGraphNode_t *pRootNodes, size_t *pNumRootNodes)

Returns a graph's root nodes.

##### Parameters

**graph**

  - Graph to query
**pRootNodes**

  - Pointer to return the root nodes
**pNumRootNodes**

  - See description

##### Returns

cudaSuccess, cudaErrorInvalidValue

##### Description

Returns a list of graph's root nodes. pRootNodes may be NULL, in which case this function will
return the number of root nodes in pNumRootNodes. Otherwise, pNumRootNodes entries will be
filled in. If pNumRootNodes is higher than the actual number of root nodes, the remaining entries
in pRootNodes will be set to NULL, and the number of nodes actually obtained will be returned in
pNumRootNodes.











See also:

cudaGraphCreate, cudaGraphGetNodes, cudaGraphGetEdges, cudaGraphNodeGetType,
cudaGraphNodeGetDependencies, cudaGraphNodeGetDependentNodes


CUDA Runtime API vRelease Version  |  386


Modules
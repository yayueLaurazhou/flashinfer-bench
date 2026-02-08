# __host__cudaError_t cudaGraphNodeGetDependencies (cudaGraphNode_t node, cudaGraphNode_t *pDependencies, cudaGraphEdgeData *edgeData, size_t *pNumDependencies)

Returns a node's dependencies.

##### Parameters

**node**

  - Node to query
**pDependencies**

  - Pointer to return the dependencies
**edgeData**

  - Optional array to return edge data for each dependency
**pNumDependencies**

  - See description


CUDA Runtime API vRelease Version  |  415


Modules

##### Returns

cudaSuccess, cudaErrorLossyQuery, cudaErrorInvalidValue

##### Description

Returns a list of node's dependencies. pDependencies may be NULL, in which case this function
will return the number of dependencies in pNumDependencies. Otherwise, pNumDependencies
entries will be filled in. If pNumDependencies is higher than the actual number of dependencies,
the remaining entries in pDependencies will be set to NULL, and the number of nodes actually
obtained will be returned in pNumDependencies.

Note that if an edge has non-zero (non-default) edge data and edgeData is NULL, this API will
return cudaErrorLossyQuery. If edgeData is non-NULL, then pDependencies must be as well.











See also:

cudaGraphNodeGetDependentNodes, cudaGraphGetNodes, cudaGraphGetRootNodes,
cudaGraphGetEdges, cudaGraphAddDependencies, cudaGraphRemoveDependencies
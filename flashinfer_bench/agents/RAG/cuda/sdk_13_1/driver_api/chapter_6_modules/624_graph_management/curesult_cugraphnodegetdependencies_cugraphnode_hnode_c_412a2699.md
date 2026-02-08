# CUresult cuGraphNodeGetDependencies (CUgraphNode hNode, CUgraphNode *dependencies, CUgraphEdgeData *edgeData, size_t *numDependencies)

Returns a node's dependencies.

###### Parameters

**hNode**

  - Node to query
**dependencies**

  - Pointer to return the dependencies
**edgeData**

  - Optional array to return edge data for each dependency
**numDependencies**

  - See description

###### Returns

CUDA_SUCCESS, CUDA_ERROR_LOSSY_QUERY, CUDA_ERROR_DEINITIALIZED,
CUDA_ERROR_NOT_INITIALIZED, CUDA_ERROR_INVALID_VALUE

###### Description

Returns a list of node's dependencies. dependencies may be NULL, in which case this function
will return the number of dependencies in numDependencies. Otherwise, numDependencies
entries will be filled in. If numDependencies is higher than the actual number of dependencies, the
remaining entries in dependencies will be set to NULL, and the number of nodes actually obtained
will be returned in numDependencies.

Note that if an edge has non-zero (non-default) edge data and edgeData is NULL, this API will
return CUDA_ERROR_LOSSY_QUERY. If edgeData is non-NULL, then dependencies must
be as well.





See also:

cuGraphNodeGetDependentNodes, cuGraphGetNodes, cuGraphGetRootNodes, cuGraphGetEdges,
cuGraphAddDependencies, cuGraphRemoveDependencies


CUDA Driver API TRM-06703-001 _vRelease Version  |  485


Modules
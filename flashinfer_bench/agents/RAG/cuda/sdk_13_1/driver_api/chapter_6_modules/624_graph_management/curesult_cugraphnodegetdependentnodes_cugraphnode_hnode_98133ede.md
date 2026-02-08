# CUresult cuGraphNodeGetDependentNodes (CUgraphNode hNode, CUgraphNode *dependentNodes, CUgraphEdgeData *edgeData, size_t *numDependentNodes)

Returns a node's dependent nodes.

###### Parameters

**hNode**

  - Node to query
**dependentNodes**

  - Pointer to return the dependent nodes
**edgeData**

  - Optional pointer to return edge data for dependent nodes
**numDependentNodes**

  - See description

###### Returns

CUDA_SUCCESS, CUDA_ERROR_LOSSY_QUERY, CUDA_ERROR_DEINITIALIZED,
CUDA_ERROR_NOT_INITIALIZED, CUDA_ERROR_INVALID_VALUE

###### Description

Returns a list of node's dependent nodes. dependentNodes may be NULL, in which case
this function will return the number of dependent nodes in numDependentNodes. Otherwise,
numDependentNodes entries will be filled in. If numDependentNodes is higher than the actual
number of dependent nodes, the remaining entries in dependentNodes will be set to NULL, and the
number of nodes actually obtained will be returned in numDependentNodes.

Note that if an edge has non-zero (non-default) edge data and edgeData is NULL, this API will
return CUDA_ERROR_LOSSY_QUERY. If edgeData is non-NULL, then dependentNodes
must be as well.





See also:

cuGraphNodeGetDependencies, cuGraphGetNodes, cuGraphGetRootNodes, cuGraphGetEdges,
cuGraphAddDependencies, cuGraphRemoveDependencies


CUDA Driver API TRM-06703-001 _vRelease Version  |  486


Modules
# CUresult cuGraphGetRootNodes (CUgraph hGraph, CUgraphNode *rootNodes, size_t *numRootNodes)

Returns a graph's root nodes.

###### Parameters

**hGraph**

  - Graph to query
**rootNodes**

  - Pointer to return the root nodes
**numRootNodes**

  - See description

###### Returns

CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
CUDA_ERROR_INVALID_VALUE

###### Description

Returns a list of hGraph's root nodes. rootNodes may be NULL, in which case this function will
return the number of root nodes in numRootNodes. Otherwise, numRootNodes entries will be


CUDA Driver API TRM-06703-001 _vRelease Version  |  468


Modules


filled in. If numRootNodes is higher than the actual number of root nodes, the remaining entries
in rootNodes will be set to NULL, and the number of nodes actually obtained will be returned in
numRootNodes.





See also:

cuGraphCreate, cuGraphGetNodes, cuGraphGetEdges, cuGraphNodeGetType,
cuGraphNodeGetDependencies, cuGraphNodeGetDependentNodes
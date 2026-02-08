# CUresult cuGraphGetNodes (CUgraph hGraph, CUgraphNode *nodes, size_t *numNodes)

Returns a graph's nodes.

###### Parameters

**hGraph**

  - Graph to query
**nodes**

  - Pointer to return the nodes
**numNodes**

  - See description


CUDA Driver API TRM-06703-001 _vRelease Version  |  467


Modules

###### Returns

CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
CUDA_ERROR_INVALID_VALUE

###### Description

Returns a list of hGraph's nodes. nodes may be NULL, in which case this function will return
the number of nodes in numNodes. Otherwise, numNodes entries will be filled in. If numNodes is
higher than the actual number of nodes, the remaining entries in nodes will be set to NULL, and the
number of nodes actually obtained will be returned in numNodes.





See also:

cuGraphCreate, cuGraphGetRootNodes, cuGraphGetEdges, cuGraphNodeGetType,
cuGraphNodeGetDependencies, cuGraphNodeGetDependentNodes
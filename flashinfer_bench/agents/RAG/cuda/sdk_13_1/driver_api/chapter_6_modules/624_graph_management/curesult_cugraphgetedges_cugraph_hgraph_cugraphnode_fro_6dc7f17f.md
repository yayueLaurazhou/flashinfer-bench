# CUresult cuGraphGetEdges (CUgraph hGraph, CUgraphNode *from, CUgraphNode *to, CUgraphEdgeData *edgeData, size_t *numEdges)

Returns a graph's dependency edges.

###### Parameters

**hGraph**

  - Graph to get the edges from
**from**

  - Location to return edge endpoints
**to**

  - Location to return edge endpoints
**edgeData**

  - Optional location to return edge data
**numEdges**

  - See description

###### Returns

CUDA_SUCCESS, CUDA_ERROR_LOSSY_QUERY, CUDA_ERROR_DEINITIALIZED,
CUDA_ERROR_NOT_INITIALIZED, CUDA_ERROR_INVALID_VALUE

###### Description

Returns a list of hGraph's dependency edges. Edges are returned via corresponding indices in from,
to and edgeData; that is, the node in to[i] has a dependency on the node in from[i] with data
edgeData[i]. from and to may both be NULL, in which case this function only returns the number
of edges in numEdges. Otherwise, numEdges entries will be filled in. If numEdges is higher
than the actual number of edges, the remaining entries in from and to will be set to NULL, and the
number of edges actually returned will be written to numEdges. edgeData may alone be NULL,
in which case the edges must all have default (zeroed) edge data. Attempting a lossy query via NULL
edgeData will result in CUDA_ERROR_LOSSY_QUERY. If edgeData is non-NULL then from
and to must be as well.





CUDA Driver API TRM-06703-001 _vRelease Version  |  466


Modules


See also:

cuGraphGetNodes, cuGraphGetRootNodes, cuGraphAddDependencies,
cuGraphRemoveDependencies, cuGraphNodeGetDependencies, cuGraphNodeGetDependentNodes
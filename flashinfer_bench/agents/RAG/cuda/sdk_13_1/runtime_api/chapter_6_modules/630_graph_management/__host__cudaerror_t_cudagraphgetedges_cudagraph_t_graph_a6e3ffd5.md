# __host__cudaError_t cudaGraphGetEdges (cudaGraph_t graph, cudaGraphNode_t *from, cudaGraphNode_t *to, cudaGraphEdgeData *edgeData, size_t *numEdges)

Returns a graph's dependency edges.

##### Parameters

**graph**

  - Graph to get the edges from
**from**

  - Location to return edge endpoints
**to**

  - Location to return edge endpoints
**edgeData**

  - Optional location to return edge data
**numEdges**

  - See description

##### Returns

cudaSuccess, cudaErrorLossyQuery, cudaErrorInvalidValue


CUDA Runtime API vRelease Version  |  383


Modules

##### Description

Returns a list of graph's dependency edges. Edges are returned via corresponding indices in from,
to and edgeData; that is, the node in to[i] has a dependency on the node in from[i] with data
edgeData[i]. from and to may both be NULL, in which case this function only returns the number
of edges in numEdges. Otherwise, numEdges entries will be filled in. If numEdges is higher
than the actual number of edges, the remaining entries in from and to will be set to NULL, and the
number of edges actually returned will be written to numEdges. edgeData may alone be NULL,
in which case the edges must all have default (zeroed) edge data. Attempting a losst query via NULL
edgeData will result in cudaErrorLossyQuery. If edgeData is non-NULL then from and to must
be as well.











See also:

cudaGraphGetNodes, cudaGraphGetRootNodes, cudaGraphAddDependencies,
cudaGraphRemoveDependencies, cudaGraphNodeGetDependencies,
cudaGraphNodeGetDependentNodes
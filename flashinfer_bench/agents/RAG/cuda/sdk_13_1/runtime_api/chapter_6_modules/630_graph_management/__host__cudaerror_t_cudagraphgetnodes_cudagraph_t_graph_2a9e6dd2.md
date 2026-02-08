# __host__cudaError_t cudaGraphGetNodes (cudaGraph_t graph, cudaGraphNode_t *nodes, size_t *numNodes)

Returns a graph's nodes.

##### Parameters

**graph**

  - Graph to query
**nodes**

  - Pointer to return the nodes
**numNodes**

  - See description

##### Returns

cudaSuccess, cudaErrorInvalidValue

##### Description

Returns a list of graph's nodes. nodes may be NULL, in which case this function will return the
number of nodes in numNodes. Otherwise, numNodes entries will be filled in. If numNodes is
higher than the actual number of nodes, the remaining entries in nodes will be set to NULL, and the
number of nodes actually obtained will be returned in numNodes.











CUDA Runtime API vRelease Version  |  385


Modules


See also:

cudaGraphCreate, cudaGraphGetRootNodes, cudaGraphGetEdges, cudaGraphNodeGetType,
cudaGraphNodeGetDependencies, cudaGraphNodeGetDependentNodes
# __host__cudaError_t cudaGraphRemoveDependencies (cudaGraph_t graph, const cudaGraphNode_t *from, const cudaGraphNode_t *to, const cudaGraphEdgeData *edgeData, size_t numDependencies)

Removes dependency edges from a graph.

##### Parameters

**graph**

  - Graph from which to remove dependencies
**from**

  - Array of nodes that provide the dependencies
**to**

  - Array of dependent nodes
**edgeData**

  - Optional array of edge data. If NULL, edge data is assumed to be default (zeroed).
**numDependencies**

  - Number of dependencies to be removed

##### Returns

cudaSuccess, cudaErrorInvalidValue

##### Description

The number of pDependencies to be removed is defined by numDependencies. Elements in
pFrom and pTo at corresponding indices define a dependency. Each node in pFrom and pTo must
belong to graph.


CUDA Runtime API vRelease Version  |  423


Modules


If numDependencies is 0, elements in pFrom and pTo will be ignored. Specifying an edge that
does not exist in the graph, with data matching edgeData, results in an error. edgeData is nullable,
which is equivalent to passing default (zeroed) data for each edge.











See also:

cudaGraphAddDependencies, cudaGraphGetEdges, cudaGraphNodeGetDependencies,
cudaGraphNodeGetDependentNodes
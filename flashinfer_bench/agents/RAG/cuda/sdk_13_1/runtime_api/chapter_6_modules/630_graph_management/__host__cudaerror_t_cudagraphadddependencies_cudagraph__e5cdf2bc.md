# __host__cudaError_t cudaGraphAddDependencies (cudaGraph_t graph, const cudaGraphNode_t *from, const cudaGraphNode_t *to, const cudaGraphEdgeData *edgeData, size_t numDependencies)

Adds dependency edges to a graph.

##### Parameters

**graph**

  - Graph to which dependencies are added


CUDA Runtime API vRelease Version  |  324


Modules


**from**

  - Array of nodes that provide the dependencies
**to**

  - Array of dependent nodes
**edgeData**

  - Optional array of edge data. If NULL, default (zeroed) edge data is assumed.
**numDependencies**

  - Number of dependencies to be added

##### Returns

cudaSuccess, cudaErrorInvalidValue

##### Description

The number of dependencies to be added is defined by numDependencies Elements in pFrom
and pTo at corresponding indices define a dependency. Each node in pFrom and pTo must belong to
graph.

If numDependencies is 0, elements in pFrom and pTo will be ignored. Specifying an existing
dependency will return an error.











See also:

cudaGraphRemoveDependencies, cudaGraphGetEdges, cudaGraphNodeGetDependencies,
cudaGraphNodeGetDependentNodes


CUDA Runtime API vRelease Version  |  325


Modules
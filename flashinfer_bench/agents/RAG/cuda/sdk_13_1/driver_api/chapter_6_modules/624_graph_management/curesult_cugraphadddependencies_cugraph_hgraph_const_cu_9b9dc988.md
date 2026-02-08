# CUresult cuGraphAddDependencies (CUgraph hGraph, const CUgraphNode *from, const CUgraphNode *to, const CUgraphEdgeData *edgeData, size_t numDependencies)

Adds dependency edges to a graph.

###### Parameters

**hGraph**

  - Graph to which dependencies are added
**from**

  - Array of nodes that provide the dependencies
**to**

  - Array of dependent nodes
**edgeData**

  - Optional array of edge data. If NULL, default (zeroed) edge data is assumed.


CUDA Driver API TRM-06703-001 _vRelease Version  |  418


Modules


**numDependencies**

  - Number of dependencies to be added

###### Returns

CUDA_SUCCESS, CUDA_ERROR_INVALID_VALUE

###### Description

The number of dependencies to be added is defined by numDependencies Elements in from and
to at corresponding indices define a dependency. Each node in from and to must belong to hGraph.

If numDependencies is 0, elements in from and to will be ignored. Specifying an existing
dependency will return an error.





See also:

cuGraphRemoveDependencies, cuGraphGetEdges, cuGraphNodeGetDependencies,
cuGraphNodeGetDependentNodes
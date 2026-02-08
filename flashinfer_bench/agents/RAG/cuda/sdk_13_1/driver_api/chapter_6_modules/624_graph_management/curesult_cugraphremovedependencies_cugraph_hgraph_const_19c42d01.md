# CUresult cuGraphRemoveDependencies (CUgraph hGraph, const CUgraphNode *from, const CUgraphNode *to, const CUgraphEdgeData *edgeData, size_t numDependencies)

Removes dependency edges from a graph.

###### Parameters

**hGraph**

  - Graph from which to remove dependencies
**from**

  - Array of nodes that provide the dependencies
**to**

  - Array of dependent nodes
**edgeData**

  - Optional array of edge data. If NULL, edge data is assumed to be default (zeroed).
**numDependencies**

  - Number of dependencies to be removed

###### Returns

CUDA_SUCCESS, CUDA_ERROR_INVALID_VALUE

###### Description

The number of dependencies to be removed is defined by numDependencies. Elements in
from and to at corresponding indices define a dependency. Each node in from and to must belong
to hGraph.


CUDA Driver API TRM-06703-001 _vRelease Version  |  492


Modules


If numDependencies is 0, elements in from and to will be ignored. Specifying an edge that does
not exist in the graph, with data matching edgeData, results in an error. edgeData is nullable,
which is equivalent to passing default (zeroed) data for each edge.

Dependencies cannot be removed from graphs which contain allocation or free nodes. Any attempt to
do so will return an error.





See also:

cuGraphAddDependencies, cuGraphGetEdges, cuGraphNodeGetDependencies,
cuGraphNodeGetDependentNodes
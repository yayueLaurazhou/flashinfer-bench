# CUresult cuGraphAddEmptyNode (CUgraphNode *phGraphNode, CUgraph hGraph, const CUgraphNode *dependencies, size_t numDependencies)

Creates an empty node and adds it to a graph.

###### Parameters

**phGraphNode**

  - Returns newly created node
**hGraph**

  - Graph to which to add the node
**dependencies**

  - Dependencies of the node
**numDependencies**

  - Number of dependencies

###### Returns

CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
CUDA_ERROR_INVALID_VALUE,


CUDA Driver API TRM-06703-001 _vRelease Version  |  419


Modules

###### Description

Creates a new node which performs no operation, and adds it to hGraph with numDependencies
dependencies specified via dependencies. It is possible for numDependencies to be 0, in which
case the node will be placed at the root of the graph. dependencies may not have any duplicate
entries. A handle to the new node will be returned in phGraphNode.

An empty node performs no operation during execution, but can be used for transitive ordering. For
example, a phased execution graph with 2 groups of n nodes with a barrier between them can be
represented using an empty node and 2*n dependency edges, rather than no empty node and n^2
dependency edges.





See also:

cuGraphAddNode, cuGraphCreate, cuGraphDestroyNode, cuGraphAddChildGraphNode,
cuGraphAddKernelNode, cuGraphAddHostNode, cuGraphAddMemcpyNode,
cuGraphAddMemsetNode
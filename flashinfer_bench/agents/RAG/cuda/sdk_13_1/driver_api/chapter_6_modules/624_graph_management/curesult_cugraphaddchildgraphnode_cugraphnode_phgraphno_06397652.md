# CUresult cuGraphAddChildGraphNode (CUgraphNode *phGraphNode, CUgraph hGraph, const CUgraphNode *dependencies, size_t numDependencies, CUgraph childGraph)

Creates a child graph node and adds it to a graph.

###### Parameters

**phGraphNode**

  - Returns newly created node
**hGraph**

  - Graph to which to add the node
**dependencies**

  - Dependencies of the node
**numDependencies**

  - Number of dependencies
**childGraph**

  - The graph to clone into this node


CUDA Driver API TRM-06703-001 _vRelease Version  |  417


Modules

###### Returns

CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
CUDA_ERROR_INVALID_VALUE,

###### Description

Creates a new node which executes an embedded graph, and adds it to hGraph with
numDependencies dependencies specified via dependencies. It is possible for
numDependencies to be 0, in which case the node will be placed at the root of the graph.
dependencies may not have any duplicate entries. A handle to the new node will be returned in
phGraphNode.

If childGraph contains allocation nodes, free nodes, or conditional nodes, this call will return an
error.

The node executes an embedded child graph. The child graph is cloned in this call.





See also:

cuGraphAddNode, cuGraphChildGraphNodeGetGraph, cuGraphCreate, cuGraphDestroyNode,
cuGraphAddEmptyNode, cuGraphAddKernelNode, cuGraphAddHostNode,
cuGraphAddMemcpyNode, cuGraphAddMemsetNode, cuGraphClone
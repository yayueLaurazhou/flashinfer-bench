# CUresult cuGraphAddHostNode (CUgraphNode *phGraphNode, CUgraph hGraph, const CUgraphNode *dependencies, size_t numDependencies, const CUDA_HOST_NODE_PARAMS *nodeParams)

Creates a host execution node and adds it to a graph.

###### Parameters

**phGraphNode**

  - Returns newly created node
**hGraph**

  - Graph to which to add the node
**dependencies**

  - Dependencies of the node
**numDependencies**

  - Number of dependencies
**nodeParams**

  - Parameters for the host node

###### Returns

CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
CUDA_ERROR_NOT_SUPPORTED, CUDA_ERROR_INVALID_VALUE

###### Description

Creates a new CPU execution node and adds it to hGraph with numDependencies dependencies
specified via dependencies and arguments specified in nodeParams. It is possible for
numDependencies to be 0, in which case the node will be placed at the root of the graph.
dependencies may not have any duplicate entries. A handle to the new node will be returned in
phGraphNode.

When the graph is launched, the node will invoke the specified CPU function. Host nodes are not
supported under MPS with pre-Volta GPUs.


CUDA Driver API TRM-06703-001 _vRelease Version  |  425


Modules





See also:

cuGraphAddNode, cuLaunchHostFunc, cuGraphHostNodeGetParams, cuGraphHostNodeSetParams,
cuGraphCreate, cuGraphDestroyNode, cuGraphAddChildGraphNode, cuGraphAddEmptyNode,
cuGraphAddKernelNode, cuGraphAddMemcpyNode, cuGraphAddMemsetNode
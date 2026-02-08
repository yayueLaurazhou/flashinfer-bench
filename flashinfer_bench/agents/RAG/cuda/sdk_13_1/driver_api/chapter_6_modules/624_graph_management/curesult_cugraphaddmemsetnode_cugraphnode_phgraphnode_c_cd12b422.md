# CUresult cuGraphAddMemsetNode (CUgraphNode *phGraphNode, CUgraph hGraph, const CUgraphNode *dependencies, size_t numDependencies, const CUDA_MEMSET_NODE_PARAMS *memsetParams, CUcontext ctx)

Creates a memset node and adds it to a graph.

###### Parameters

**phGraphNode**

  - Returns newly created node
**hGraph**

  - Graph to which to add the node
**dependencies**

  - Dependencies of the node
**numDependencies**

  - Number of dependencies
**memsetParams**

  - Parameters for the memory set
**ctx**

  - Context on which to run the node

###### Returns

CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
CUDA_ERROR_INVALID_VALUE, CUDA_ERROR_INVALID_CONTEXT

###### Description

Creates a new memset node and adds it to hGraph with numDependencies dependencies specified
via dependencies. It is possible for numDependencies to be 0, in which case the node will be
placed at the root of the graph. dependencies may not have any duplicate entries. A handle to the
new node will be returned in phGraphNode.

The element size must be 1, 2, or 4 bytes. When the graph is launched, the node will perform the
memset described by memsetParams.





CUDA Driver API TRM-06703-001 _vRelease Version  |  433


Modules


See also:

cuGraphAddNode, cuMemsetD2D32, cuGraphMemsetNodeGetParams,
cuGraphMemsetNodeSetParams, cuGraphCreate, cuGraphDestroyNode,
cuGraphAddChildGraphNode, cuGraphAddEmptyNode, cuGraphAddKernelNode,
cuGraphAddHostNode, cuGraphAddMemcpyNode
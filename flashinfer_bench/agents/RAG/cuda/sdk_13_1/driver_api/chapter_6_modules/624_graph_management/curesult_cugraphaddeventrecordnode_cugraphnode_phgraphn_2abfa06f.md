# CUresult cuGraphAddEventRecordNode (CUgraphNode *phGraphNode, CUgraph hGraph, const CUgraphNode *dependencies, size_t numDependencies, CUevent event)

Creates an event record node and adds it to a graph.

###### Parameters

**phGraphNode**

  - Returns newly created node
**hGraph**

  - Graph to which to add the node
**dependencies**

  - Dependencies of the node
**numDependencies**

  - Number of dependencies
**event**

  - Event for the node


CUDA Driver API TRM-06703-001 _vRelease Version  |  420


Modules

###### Returns

CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
CUDA_ERROR_NOT_SUPPORTED, CUDA_ERROR_INVALID_VALUE

###### Description

Creates a new event record node and adds it to hGraph with numDependencies dependencies
specified via dependencies and event specified in event. It is possible for numDependencies
to be 0, in which case the node will be placed at the root of the graph. dependencies may not have
any duplicate entries. A handle to the new node will be returned in phGraphNode.

Each launch of the graph will record event to capture execution of the node's dependencies.





See also:

cuGraphAddNode, cuGraphAddEventWaitNode, cuEventRecordWithFlags, cuStreamWaitEvent,
cuGraphCreate, cuGraphDestroyNode, cuGraphAddChildGraphNode, cuGraphAddEmptyNode,
cuGraphAddKernelNode, cuGraphAddMemcpyNode, cuGraphAddMemsetNode
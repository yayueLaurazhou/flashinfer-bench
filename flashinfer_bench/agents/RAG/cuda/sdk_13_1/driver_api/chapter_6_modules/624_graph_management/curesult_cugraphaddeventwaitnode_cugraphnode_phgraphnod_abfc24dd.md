# CUresult cuGraphAddEventWaitNode (CUgraphNode *phGraphNode, CUgraph hGraph, const CUgraphNode *dependencies, size_t numDependencies, CUevent event)

Creates an event wait node and adds it to a graph.

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


CUDA Driver API TRM-06703-001 _vRelease Version  |  421


Modules

###### Returns

CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
CUDA_ERROR_NOT_SUPPORTED, CUDA_ERROR_INVALID_VALUE

###### Description

Creates a new event wait node and adds it to hGraph with numDependencies dependencies
specified via dependencies and event specified in event. It is possible for numDependencies
to be 0, in which case the node will be placed at the root of the graph. dependencies may not have
any duplicate entries. A handle to the new node will be returned in phGraphNode.

The graph node will wait for all work captured in event. See cuEventRecord() for details on what is
captured by an event. event may be from a different context or device than the launch stream.





See also:

cuGraphAddNode, cuGraphAddEventRecordNode, cuEventRecordWithFlags, cuStreamWaitEvent,
cuGraphCreate, cuGraphDestroyNode, cuGraphAddChildGraphNode, cuGraphAddEmptyNode,
cuGraphAddKernelNode, cuGraphAddMemcpyNode, cuGraphAddMemsetNode
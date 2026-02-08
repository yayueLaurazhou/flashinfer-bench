# CUresult cuGraphAddExternalSemaphoresSignalNode (CUgraphNode *phGraphNode, CUgraph hGraph, const CUgraphNode *dependencies, size_t numDependencies, const CUDA_EXT_SEM_SIGNAL_NODE_PARAMS *nodeParams)

Creates an external semaphore signal node and adds it to a graph.

###### Parameters

**phGraphNode**

  - Returns newly created node
**hGraph**

  - Graph to which to add the node
**dependencies**

  - Dependencies of the node


CUDA Driver API TRM-06703-001 _vRelease Version  |  422


Modules


**numDependencies**

  - Number of dependencies
**nodeParams**

  - Parameters for the node

###### Returns

CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
CUDA_ERROR_NOT_SUPPORTED, CUDA_ERROR_INVALID_VALUE

###### Description

Creates a new external semaphore signal node and adds it to hGraph with numDependencies
dependencies specified via dependencies and arguments specified in nodeParams. It is possible
for numDependencies to be 0, in which case the node will be placed at the root of the graph.
dependencies may not have any duplicate entries. A handle to the new node will be returned in
phGraphNode.

Performs a signal operation on a set of externally allocated semaphore objects when the node is
launched. The operation(s) will occur after all of the node's dependencies have completed.





See also:

cuGraphAddNode, cuGraphExternalSemaphoresSignalNodeGetParams,
cuGraphExternalSemaphoresSignalNodeSetParams,
cuGraphExecExternalSemaphoresSignalNodeSetParams, cuGraphAddExternalSemaphoresWaitNode,
cuImportExternalSemaphore, cuSignalExternalSemaphoresAsync, cuWaitExternalSemaphoresAsync,
cuGraphCreate, cuGraphDestroyNode, cuGraphAddEventRecordNode, cuGraphAddEventWaitNode,
cuGraphAddChildGraphNode, cuGraphAddEmptyNode, cuGraphAddKernelNode,
cuGraphAddMemcpyNode, cuGraphAddMemsetNode


CUDA Driver API TRM-06703-001 _vRelease Version  |  423


Modules
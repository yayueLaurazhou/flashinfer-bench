# CUresult cuGraphAddMemFreeNode (CUgraphNode *phGraphNode, CUgraph hGraph, const CUgraphNode *dependencies, size_t numDependencies, CUdeviceptr dptr)

Creates a memory free node and adds it to a graph.

###### Parameters

**phGraphNode**

  - Returns newly created node
**hGraph**

  - Graph to which to add the node
**dependencies**

  - Dependencies of the node
**numDependencies**

  - Number of dependencies
**dptr**

  - Address of memory to free


CUDA Driver API TRM-06703-001 _vRelease Version  |  431


Modules

###### Returns

CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
CUDA_ERROR_NOT_SUPPORTED, CUDA_ERROR_INVALID_VALUE

###### Description

Creates a new memory free node and adds it to hGraph with numDependencies dependencies
specified via dependencies and arguments specified in nodeParams. It is possible for
numDependencies to be 0, in which case the node will be placed at the root of the graph.
dependencies may not have any duplicate entries. A handle to the new node will be returned in
phGraphNode.

cuGraphAddMemFreeNode will return CUDA_ERROR_INVALID_VALUE if the user attempts to
free:

an allocation twice in the same graph.

###### **‣**

an address that was not returned by an allocation node.

###### **‣**

an invalid address.

###### **‣**

The following restrictions apply to graphs which contain allocation and/or memory free nodes:

Nodes and edges of the graph cannot be deleted.

###### **‣**

The graph can only be used in a child node if the ownership is moved to the parent.

###### **‣**

Only one instantiation of the graph may exist at any point in time.

###### **‣**

The graph cannot be cloned.

###### **‣**

See also:

cuGraphAddNode, cuGraphAddMemAllocNode, cuGraphMemFreeNodeGetParams,
cuDeviceGraphMemTrim, cuDeviceGetGraphMemAttribute, cuDeviceSetGraphMemAttribute,
cuMemAllocAsync, cuMemFreeAsync, cuGraphCreate, cuGraphDestroyNode,
cuGraphAddChildGraphNode, cuGraphAddEmptyNode, cuGraphAddEventRecordNode,
cuGraphAddEventWaitNode, cuGraphAddExternalSemaphoresSignalNode,
cuGraphAddExternalSemaphoresWaitNode, cuGraphAddKernelNode, cuGraphAddMemcpyNode,
cuGraphAddMemsetNode


CUDA Driver API TRM-06703-001 _vRelease Version  |  432


Modules
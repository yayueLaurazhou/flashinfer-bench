# CUresult cuGraphAddMemAllocNode (CUgraphNode *phGraphNode, CUgraph hGraph, const CUgraphNode *dependencies, size_t numDependencies, CUDA_MEM_ALLOC_NODE_PARAMS *nodeParams)

Creates an allocation node and adds it to a graph.

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

  - Parameters for the node


CUDA Driver API TRM-06703-001 _vRelease Version  |  428


Modules

###### Returns

CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
CUDA_ERROR_NOT_SUPPORTED, CUDA_ERROR_INVALID_VALUE

###### Description

Creates a new allocation node and adds it to hGraph with numDependencies dependencies
specified via dependencies and arguments specified in nodeParams. It is possible for
numDependencies to be 0, in which case the node will be placed at the root of the graph.
dependencies may not have any duplicate entries. A handle to the new node will be returned in
phGraphNode.

When cuGraphAddMemAllocNode creates an allocation node, it returns the address of the allocation in
nodeParams.dptr. The allocation's address remains fixed across instantiations and launches.

If the allocation is freed in the same graph, by creating a free node using cuGraphAddMemFreeNode,
the allocation can be accessed by nodes ordered after the allocation node but before the free node.
These allocations cannot be freed outside the owning graph, and they can only be freed once in the
owning graph.

If the allocation is not freed in the same graph, then it can be accessed not only by nodes in the graph
which are ordered after the allocation node, but also by stream operations ordered after the graph's
execution but before the allocation is freed.

Allocations which are not freed in the same graph can be freed by:

passing the allocation to cuMemFreeAsync or cuMemFree;

###### **‣**

launching a graph with a free node for that allocation; or

###### **‣**

specifying CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH during

###### **‣**

instantiation, which makes each launch behave as though it called cuMemFreeAsync for every
unfreed allocation.

It is not possible to free an allocation in both the owning graph and another graph. If the allocation
is freed in the same graph, a free node cannot be added to another graph. If the allocation is freed in
another graph, a free node can no longer be added to the owning graph.

The following restrictions apply to graphs which contain allocation and/or memory free nodes:

Nodes and edges of the graph cannot be deleted.

###### **‣**

The graph can only be used in a child node if the ownership is moved to the parent.

###### **‣**

Only one instantiation of the graph may exist at any point in time.

###### **‣**

The graph cannot be cloned.

###### **‣**

CUDA Driver API TRM-06703-001 _vRelease Version  |  429


Modules


**‣** Note that this function may also return error codes from previous, asynchronous launches.


See also:

cuGraphAddNode, cuGraphAddMemFreeNode, cuGraphMemAllocNodeGetParams,
cuDeviceGraphMemTrim, cuDeviceGetGraphMemAttribute, cuDeviceSetGraphMemAttribute,
cuMemAllocAsync, cuMemFreeAsync, cuGraphCreate, cuGraphDestroyNode,
cuGraphAddChildGraphNode, cuGraphAddEmptyNode, cuGraphAddEventRecordNode,
cuGraphAddEventWaitNode, cuGraphAddExternalSemaphoresSignalNode,
cuGraphAddExternalSemaphoresWaitNode, cuGraphAddKernelNode, cuGraphAddMemcpyNode,
cuGraphAddMemsetNode
# __host__cudaError_t cudaGraphAddMemAllocNode (cudaGraphNode_t *pGraphNode, cudaGraph_t graph, const cudaGraphNode_t *pDependencies, size_t numDependencies, cudaMemAllocNodeParams *nodeParams)

Creates an allocation node and adds it to a graph.

##### Parameters

**pGraphNode**

  - Returns newly created node
**graph**

  - Graph to which to add the node
**pDependencies**

  - Dependencies of the node
**numDependencies**

  - Number of dependencies


CUDA Runtime API vRelease Version  |  335


Modules


**nodeParams**

  - Parameters for the node

##### Returns

cudaSuccess, cudaErrorCudartUnloading, cudaErrorInitializationError, cudaErrorNotSupported,
cudaErrorInvalidValue, cudaErrorOutOfMemory

##### Description

Creates a new allocation node and adds it to graph with numDependencies dependencies
specified via pDependencies and arguments specified in nodeParams. It is possible for
numDependencies to be 0, in which case the node will be placed at the root of the graph.
pDependencies may not have any duplicate entries. A handle to the new node will be returned in
pGraphNode.

When cudaGraphAddMemAllocNode creates an allocation node, it returns the address of the allocation
in nodeParams.dptr. The allocation's address remains fixed across instantiations and launches.

If the allocation is freed in the same graph, by creating a free node using
cudaGraphAddMemFreeNode, the allocation can be accessed by nodes ordered after the allocation
node but before the free node. These allocations cannot be freed outside the owning graph, and they
can only be freed once in the owning graph.

If the allocation is not freed in the same graph, then it can be accessed not only by nodes in the graph
which are ordered after the allocation node, but also by stream operations ordered after the graph's
execution but before the allocation is freed.

Allocations which are not freed in the same graph can be freed by:

passing the allocation to cudaMemFreeAsync or cudaMemFree;

##### **‣**

launching a graph with a free node for that allocation; or

##### **‣**

specifying cudaGraphInstantiateFlagAutoFreeOnLaunch during instantiation, which makes each

##### **‣**

launch behave as though it called cudaMemFreeAsync for every unfreed allocation.

It is not possible to free an allocation in both the owning graph and another graph. If the allocation
is freed in the same graph, a free node cannot be added to another graph. If the allocation is freed in
another graph, a free node can no longer be added to the owning graph.

The following restrictions apply to graphs which contain allocation and/or memory free nodes:

Nodes and edges of the graph cannot be deleted.

##### **‣**

The graph can only be used in a child node if the ownership is moved to the parent.

##### **‣**

Only one instantiation of the graph may exist at any point in time.

##### **‣**

The graph cannot be cloned.

##### **‣**

Note:


CUDA Runtime API vRelease Version  |  336


Modules







See also:

cudaGraphAddNode, cudaGraphAddMemFreeNode, cudaGraphMemAllocNodeGetParams,
cudaDeviceGraphMemTrim, cudaDeviceGetGraphMemAttribute, cudaDeviceSetGraphMemAttribute,
cudaMallocAsync, cudaFreeAsync, cudaGraphCreate, cudaGraphDestroyNode,
cudaGraphAddChildGraphNode, cudaGraphAddEmptyNode, cudaGraphAddEventRecordNode,
cudaGraphAddEventWaitNode, cudaGraphAddExternalSemaphoresSignalNode,
cudaGraphAddExternalSemaphoresWaitNode, cudaGraphAddKernelNode,
cudaGraphAddMemcpyNode, cudaGraphAddMemsetNode
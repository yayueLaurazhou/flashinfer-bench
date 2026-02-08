# __host__cudaError_t cudaGraphAddMemFreeNode (cudaGraphNode_t *pGraphNode, cudaGraph_t graph, const cudaGraphNode_t *pDependencies, size_t numDependencies, void *dptr)

Creates a memory free node and adds it to a graph.

##### Parameters

**pGraphNode**

  - Returns newly created node
**graph**

  - Graph to which to add the node
**pDependencies**

  - Dependencies of the node


CUDA Runtime API vRelease Version  |  343


Modules


**numDependencies**

  - Number of dependencies
**dptr**

  - Address of memory to free

##### Returns

cudaSuccess, cudaErrorCudartUnloading, cudaErrorInitializationError, cudaErrorNotSupported,
cudaErrorInvalidValue, cudaErrorOutOfMemory

##### Description

Creates a new memory free node and adds it to graph with numDependencies
dependencies specified via pDependencies and address specified in dptr. It is possible for
numDependencies to be 0, in which case the node will be placed at the root of the graph.
pDependencies may not have any duplicate entries. A handle to the new node will be returned in
pGraphNode.

cudaGraphAddMemFreeNode will return cudaErrorInvalidValue if the user attempts to free:

an allocation twice in the same graph.

##### **‣**

an address that was not returned by an allocation node.

##### **‣**

an invalid address.

##### **‣**

The following restrictions apply to graphs which contain allocation and/or memory free nodes:

Nodes and edges of the graph cannot be deleted.

##### **‣**

The graph can only be used in a child node if the ownership is moved to the parent.

##### **‣**

Only one instantiation of the graph may exist at any point in time.

##### **‣**

The graph cannot be cloned.

##### **‣**

See also:

cudaGraphAddNode, cudaGraphAddMemAllocNode, cudaGraphMemFreeNodeGetParams,
cudaDeviceGraphMemTrim, cudaDeviceGetGraphMemAttribute, cudaDeviceSetGraphMemAttribute,
cudaMallocAsync, cudaFreeAsync, cudaGraphCreate, cudaGraphDestroyNode,
cudaGraphAddChildGraphNode, cudaGraphAddEmptyNode, cudaGraphAddEventRecordNode,
cudaGraphAddEventWaitNode, cudaGraphAddExternalSemaphoresSignalNode,
cudaGraphAddExternalSemaphoresWaitNode, cudaGraphAddKernelNode,
cudaGraphAddMemcpyNode, cudaGraphAddMemsetNode


CUDA Runtime API vRelease Version  |  344


Modules
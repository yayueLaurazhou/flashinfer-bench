# __host__cudaError_t cudaGraphAddExternalSemaphoresWaitNode (cudaGraphNode_t *pGraphNode, cudaGraph_t graph, const cudaGraphNode_t *pDependencies, size_t numDependencies, const cudaExternalSemaphoreWaitNodeParams *nodeParams)

Creates an external semaphore wait node and adds it to a graph.

##### Parameters

**pGraphNode**

  - Returns newly created node
**graph**

  - Graph to which to add the node


CUDA Runtime API vRelease Version  |  330


Modules


**pDependencies**

  - Dependencies of the node
**numDependencies**

  - Number of dependencies
**nodeParams**

  - Parameters for the node

##### Returns

cudaSuccess, cudaErrorInvalidValue

##### Description

Creates a new external semaphore wait node and adds it to graph with numDependencies
dependencies specified via dependencies and arguments specified in nodeParams. It is possible
for numDependencies to be 0, in which case the node will be placed at the root of the graph.
dependencies may not have any duplicate entries. A handle to the new node will be returned in
pGraphNode.

Performs a wait operation on a set of externally allocated semaphore objects when the node is
launched. The node's dependencies will not be launched until the wait operation has completed.











See also:

cudaGraphAddNode, cudaGraphExternalSemaphoresWaitNodeGetParams,
cudaGraphExternalSemaphoresWaitNodeSetParams,
cudaGraphExecExternalSemaphoresWaitNodeSetParams,
cudaGraphAddExternalSemaphoresSignalNode, cudaImportExternalSemaphore,
cudaSignalExternalSemaphoresAsync, cudaWaitExternalSemaphoresAsync, cudaGraphCreate,
cudaGraphDestroyNode, cudaGraphAddEventRecordNode, cudaGraphAddEventWaitNode,
cudaGraphAddChildGraphNode, cudaGraphAddEmptyNode, cudaGraphAddKernelNode,
cudaGraphAddMemcpyNode, cudaGraphAddMemsetNode


CUDA Runtime API vRelease Version  |  331


Modules
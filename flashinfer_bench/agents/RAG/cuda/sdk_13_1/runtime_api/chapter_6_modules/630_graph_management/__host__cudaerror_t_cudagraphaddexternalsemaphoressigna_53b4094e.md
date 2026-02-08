# __host__cudaError_t cudaGraphAddExternalSemaphoresSignalNode (cudaGraphNode_t *pGraphNode, cudaGraph_t graph, const cudaGraphNode_t *pDependencies, size_t numDependencies, const cudaExternalSemaphoreSignalNodeParams *nodeParams)

Creates an external semaphore signal node and adds it to a graph.

##### Parameters

**pGraphNode**

  - Returns newly created node
**graph**

  - Graph to which to add the node
**pDependencies**

  - Dependencies of the node
**numDependencies**

  - Number of dependencies
**nodeParams**

  - Parameters for the node

##### Returns

cudaSuccess, cudaErrorInvalidValue

##### Description

Creates a new external semaphore signal node and adds it to graph with numDependencies
dependencies specified via dependencies and arguments specified in nodeParams. It is possible
for numDependencies to be 0, in which case the node will be placed at the root of the graph.


CUDA Runtime API vRelease Version  |  329


Modules


dependencies may not have any duplicate entries. A handle to the new node will be returned in
pGraphNode.

Performs a signal operation on a set of externally allocated semaphore objects when the node is
launched. The operation(s) will occur after all of the node's dependencies have completed.











See also:

cudaGraphAddNode, cudaGraphExternalSemaphoresSignalNodeGetParams,
cudaGraphExternalSemaphoresSignalNodeSetParams,
cudaGraphExecExternalSemaphoresSignalNodeSetParams,
cudaGraphAddExternalSemaphoresWaitNode, cudaImportExternalSemaphore,
cudaSignalExternalSemaphoresAsync, cudaWaitExternalSemaphoresAsync, cudaGraphCreate,
cudaGraphDestroyNode, cudaGraphAddEventRecordNode, cudaGraphAddEventWaitNode,
cudaGraphAddChildGraphNode, cudaGraphAddEmptyNode, cudaGraphAddKernelNode,
cudaGraphAddMemcpyNode, cudaGraphAddMemsetNode
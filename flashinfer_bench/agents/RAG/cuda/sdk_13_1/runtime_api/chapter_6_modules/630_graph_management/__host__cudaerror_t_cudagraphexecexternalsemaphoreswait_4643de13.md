# __host__cudaError_t cudaGraphExecExternalSemaphoresWaitNodeSetParams (cudaGraphExec_t hGraphExec, cudaGraphNode_t hNode, const cudaExternalSemaphoreWaitNodeParams *nodeParams)

Sets the parameters for an external semaphore wait node in the given graphExec.

##### Parameters

**hGraphExec**

  - The executable graph in which to set the specified node
**hNode**

  - semaphore wait node from the graph from which graphExec was instantiated
**nodeParams**

  - Updated Parameters to set

##### Returns

cudaSuccess, cudaErrorInvalidValue,

##### Description

Sets the parameters of an external semaphore wait node in an executable graph hGraphExec. The
node is identified by the corresponding node hNode in the non-executable graph, from which the
executable graph was instantiated.

hNode must not have been removed from the original graph.

The modifications only affect future launches of hGraphExec. Already enqueued or running
launches of hGraphExec are not affected by this call. hNode is also not modified by this call.

Changing nodeParams->numExtSems is not supported.











CUDA Runtime API vRelease Version  |  363


Modules


See also:

cudaGraphExecNodeSetParams, cudaGraphAddExternalSemaphoresWaitNode,
cudaImportExternalSemaphore, cudaSignalExternalSemaphoresAsync,
cudaWaitExternalSemaphoresAsync, cudaGraphExecKernelNodeSetParams,
cudaGraphExecMemcpyNodeSetParams, cudaGraphExecMemsetNodeSetParams,
cudaGraphExecHostNodeSetParams, cudaGraphExecChildGraphNodeSetParams,
cudaGraphExecEventRecordNodeSetEvent, cudaGraphExecEventWaitNodeSetEvent,
cudaGraphExecExternalSemaphoresSignalNodeSetParams, cudaGraphExecUpdate,
cudaGraphInstantiate
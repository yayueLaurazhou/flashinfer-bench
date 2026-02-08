# __host__cudaError_t cudaGraphExecHostNodeSetParams (cudaGraphExec_t hGraphExec, cudaGraphNode_t node, const cudaHostNodeParams *pNodeParams)

Sets the parameters for a host node in the given graphExec.

##### Parameters

**hGraphExec**

  - The executable graph in which to set the specified node
**node**

  - Host node from the graph which was used to instantiate graphExec
**pNodeParams**

  - Updated Parameters to set

##### Returns

cudaSuccess, cudaErrorInvalidValue,


CUDA Runtime API vRelease Version  |  365


Modules

##### Description

Updates the work represented by node in hGraphExec as though node had contained
pNodeParams at instantiation. node must remain in the graph which was used to instantiate
hGraphExec. Changed edges to and from node are ignored.

The modifications only affect future launches of hGraphExec. Already enqueued or running
launches of hGraphExec are not affected by this call. node is also not modified by this call.











See also:

cudaGraphExecNodeSetParams, cudaGraphAddHostNode, cudaGraphHostNodeSetParams,
cudaGraphExecKernelNodeSetParams, cudaGraphExecMemcpyNodeSetParams,
cudaGraphExecMemsetNodeSetParams, cudaGraphExecChildGraphNodeSetParams,
cudaGraphExecEventRecordNodeSetEvent, cudaGraphExecEventWaitNodeSetEvent,
cudaGraphExecExternalSemaphoresSignalNodeSetParams,
cudaGraphExecExternalSemaphoresWaitNodeSetParams, cudaGraphExecUpdate,
cudaGraphInstantiate
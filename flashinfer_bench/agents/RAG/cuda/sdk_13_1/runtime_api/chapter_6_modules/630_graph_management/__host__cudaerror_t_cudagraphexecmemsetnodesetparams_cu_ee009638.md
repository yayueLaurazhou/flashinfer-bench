# __host__cudaError_t cudaGraphExecMemsetNodeSetParams (cudaGraphExec_t hGraphExec, cudaGraphNode_t node, const cudaMemsetParams *pNodeParams)

Sets the parameters for a memset node in the given graphExec.

##### Parameters

**hGraphExec**

  - The executable graph in which to set the specified node
**node**

  - Memset node from the graph which was used to instantiate graphExec
**pNodeParams**

  - Updated Parameters to set

##### Returns

cudaSuccess, cudaErrorInvalidValue,

##### Description

Updates the work represented by node in hGraphExec as though node had contained
pNodeParams at instantiation. node must remain in the graph which was used to instantiate
hGraphExec. Changed edges to and from node are ignored.

Zero sized operations are not supported.

The new destination pointer in pNodeParams must be to the same kind of allocation as the original
destination pointer and have the same context association and device mapping as the original
destination pointer.

Both the value and pointer address may be updated. Changing other aspects of the memset (width,
height, element size or pitch) may cause the update to be rejected. Specifically, for 2d memsets, all
dimension changes are rejected. For 1d memsets, changes in height are explicitly rejected and other
changes are opportunistically allowed if the resulting work maps onto the work resources already
allocated for the node.

The modifications only affect future launches of hGraphExec. Already enqueued or running
launches of hGraphExec are not affected by this call. node is also not modified by this call.


CUDA Runtime API vRelease Version  |  374


Modules











See also:

cudaGraphExecNodeSetParams, cudaGraphAddMemsetNode,
cudaGraphMemsetNodeSetParams, cudaGraphExecKernelNodeSetParams,
cudaGraphExecMemcpyNodeSetParams, cudaGraphExecHostNodeSetParams,
cudaGraphExecChildGraphNodeSetParams, cudaGraphExecEventRecordNodeSetEvent,
cudaGraphExecEventWaitNodeSetEvent, cudaGraphExecExternalSemaphoresSignalNodeSetParams,
cudaGraphExecExternalSemaphoresWaitNodeSetParams, cudaGraphExecUpdate,
cudaGraphInstantiate
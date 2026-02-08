# __host__cudaError_t cudaGraphExecMemcpyNodeSetParams (cudaGraphExec_t hGraphExec, cudaGraphNode_t node, const cudaMemcpy3DParms *pNodeParams)

Sets the parameters for a memcpy node in the given graphExec.

##### Parameters

**hGraphExec**

  - The executable graph in which to set the specified node
**node**

  - Memcpy node from the graph which was used to instantiate graphExec
**pNodeParams**

  - Updated Parameters to set

##### Returns

cudaSuccess, cudaErrorInvalidValue,

##### Description

Updates the work represented by node in hGraphExec as though node had contained
pNodeParams at instantiation. node must remain in the graph which was used to instantiate
hGraphExec. Changed edges to and from node are ignored.

The source and destination memory in pNodeParams must be allocated from the same contexts as
the original source and destination memory. Both the instantiation-time memory operands and the
memory operands in pNodeParams must be 1-dimensional. Zero-length operations are not supported.


CUDA Runtime API vRelease Version  |  368


Modules


The modifications only affect future launches of hGraphExec. Already enqueued or running
launches of hGraphExec are not affected by this call. node is also not modified by this call.

Returns cudaErrorInvalidValue if the memory operands' mappings changed or either the original or
new memory operands are multidimensional.











See also:

cudaGraphExecNodeSetParams, cudaGraphAddMemcpyNode,
cudaGraphMemcpyNodeSetParams, cudaGraphExecMemcpyNodeSetParamsToSymbol,
cudaGraphExecMemcpyNodeSetParamsFromSymbol, cudaGraphExecMemcpyNodeSetParams1D,
cudaGraphExecKernelNodeSetParams, cudaGraphExecMemsetNodeSetParams,
cudaGraphExecHostNodeSetParams, cudaGraphExecChildGraphNodeSetParams,
cudaGraphExecEventRecordNodeSetEvent, cudaGraphExecEventWaitNodeSetEvent,
cudaGraphExecExternalSemaphoresSignalNodeSetParams,
cudaGraphExecExternalSemaphoresWaitNodeSetParams, cudaGraphExecUpdate,
cudaGraphInstantiate
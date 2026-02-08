# __host__cudaError_t cudaGraphExecMemcpyNodeSetParams1D (cudaGraphExec_t hGraphExec, cudaGraphNode_t node, void *dst, const void *src, size_t count, cudaMemcpyKind kind)

Sets the parameters for a memcpy node in the given graphExec to perform a 1-dimensional copy.

##### Parameters

**hGraphExec**

  - The executable graph in which to set the specified node
**node**

  - Memcpy node from the graph which was used to instantiate graphExec


CUDA Runtime API vRelease Version  |  369


Modules


**dst**

  - Destination memory address
**src**

  - Source memory address
**count**

  - Size in bytes to copy
**kind**

  - Type of transfer

##### Returns

cudaSuccess, cudaErrorInvalidValue

##### Description

Updates the work represented by node in hGraphExec as though node had contained the given
params at instantiation. node must remain in the graph which was used to instantiate hGraphExec.
Changed edges to and from node are ignored.

src and dst must be allocated from the same contexts as the original source and destination memory.
The instantiation-time memory operands must be 1-dimensional. Zero-length operations are not
supported.

The modifications only affect future launches of hGraphExec. Already enqueued or running
launches of hGraphExec are not affected by this call. node is also not modified by this call.

Returns cudaErrorInvalidValue if the memory operands' mappings changed or the original memory
operands are multidimensional.











See also:

cudaGraphAddMemcpyNode, cudaGraphAddMemcpyNode1D,
cudaGraphMemcpyNodeSetParams, cudaGraphMemcpyNodeSetParams1D,
cudaGraphExecMemcpyNodeSetParams, cudaGraphExecKernelNodeSetParams,
cudaGraphExecMemsetNodeSetParams, cudaGraphExecHostNodeSetParams,
cudaGraphExecChildGraphNodeSetParams, cudaGraphExecEventRecordNodeSetEvent,
cudaGraphExecEventWaitNodeSetEvent, cudaGraphExecExternalSemaphoresSignalNodeSetParams,


CUDA Runtime API vRelease Version  |  370


Modules


cudaGraphExecExternalSemaphoresWaitNodeSetParams, cudaGraphExecUpdate,
cudaGraphInstantiate
# template < class T > __host__cudaError_t cudaGraphExecMemcpyNodeSetParamsToSymbol (cudaGraphExec_t hGraphExec, cudaGraphNode_t node, const T symbol, const void *src, size_t count, size_t offset, cudaMemcpyKind kind)

Sets the parameters for a memcpy node in the given graphExec to copy to a symbol on the device.

##### Parameters

**hGraphExec**

  - The executable graph in which to set the specified node
**node**

  - Memcpy node from the graph which was used to instantiate graphExec
**symbol**

  - Device symbol address
**src**

  - Source memory address
**count**

  - Size in bytes to copy
**offset**

  - Offset from start of symbol in bytes
**kind**

  - Type of transfer

##### Returns

cudaSuccess, cudaErrorInvalidValue

##### Description

Updates the work represented by node in hGraphExec as though node had contained the given
params at instantiation. node must remain in the graph which was used to instantiate hGraphExec.
Changed edges to and from node are ignored.

src and symbol must be allocated from the same contexts as the original source and destination
memory. The instantiation-time memory operands must be 1-dimensional. Zero-length operations are
not supported.

The modifications only affect future launches of hGraphExec. Already enqueued or running
launches of hGraphExec are not affected by this call. node is also not modified by this call.


CUDA Runtime API vRelease Version  |  476


Modules


Returns cudaErrorInvalidValue if the memory operands' mappings changed or the original memory
operands are multidimensional.











See also:

cudaGraphAddMemcpyNode, cudaGraphAddMemcpyNodeToSymbol,
cudaGraphMemcpyNodeSetParams, cudaGraphMemcpyNodeSetParamsToSymbol,
cudaGraphInstantiate, cudaGraphExecMemcpyNodeSetParams,
cudaGraphExecMemcpyNodeSetParamsFromSymbol, cudaGraphExecKernelNodeSetParams,
cudaGraphExecMemsetNodeSetParams, cudaGraphExecHostNodeSetParams
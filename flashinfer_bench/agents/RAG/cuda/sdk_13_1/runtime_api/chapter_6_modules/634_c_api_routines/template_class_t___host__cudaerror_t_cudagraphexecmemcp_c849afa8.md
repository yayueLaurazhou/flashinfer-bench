# template < class T > __host__cudaError_t cudaGraphExecMemcpyNodeSetParamsFromSymbol (cudaGraphExec_t hGraphExec, cudaGraphNode_t node, void *dst, const T symbol, size_t count, size_t offset, cudaMemcpyKind kind)

Sets the parameters for a memcpy node in the given graphExec to copy from a symbol on the device.

##### Parameters

**hGraphExec**

  - The executable graph in which to set the specified node
**node**

  - Memcpy node from the graph which was used to instantiate graphExec


CUDA Runtime API vRelease Version  |  474


Modules


**dst**

  - Destination memory address
**symbol**

  - Device symbol address
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

symbol and dst must be allocated from the same contexts as the original source and destination
memory. The instantiation-time memory operands must be 1-dimensional. Zero-length operations are
not supported.

The modifications only affect future launches of hGraphExec. Already enqueued or running
launches of hGraphExec are not affected by this call. node is also not modified by this call.

Returns cudaErrorInvalidValue if the memory operands' mappings changed or the original memory
operands are multidimensional.











See also:

cudaGraphAddMemcpyNode, cudaGraphAddMemcpyNodeFromSymbol,
cudaGraphMemcpyNodeSetParams, cudaGraphMemcpyNodeSetParamsFromSymbol,
cudaGraphInstantiate, cudaGraphExecMemcpyNodeSetParams,


CUDA Runtime API vRelease Version  |  475


Modules


cudaGraphExecMemcpyNodeSetParamsToSymbol, cudaGraphExecKernelNodeSetParams,
cudaGraphExecMemsetNodeSetParams, cudaGraphExecHostNodeSetParams
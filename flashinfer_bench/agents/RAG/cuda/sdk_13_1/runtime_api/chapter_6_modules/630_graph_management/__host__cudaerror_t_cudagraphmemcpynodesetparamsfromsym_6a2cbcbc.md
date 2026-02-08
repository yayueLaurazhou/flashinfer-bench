# __host__cudaError_t cudaGraphMemcpyNodeSetParamsFromSymbol (cudaGraphNode_t node, void *dst, const void *symbol, size_t count, size_t offset, cudaMemcpyKind kind)

Sets a memcpy node's parameters to copy from a symbol on the device.

##### Parameters

**node**

  - Node to set the parameters for
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

Sets the parameters of memcpy node node to the copy described by the provided parameters.

When the graph is launched, the node will copy count bytes from the memory area pointed to by
offset bytes from the start of symbol symbol to the memory area pointed to by dst. The memory
areas may not overlap. symbol is a variable that resides in global or constant memory space. kind
can be either cudaMemcpyDeviceToHost, cudaMemcpyDeviceToDevice, or cudaMemcpyDefault.
Passing cudaMemcpyDefault is recommended, in which case the type of transfer is inferred from the
pointer values. However, cudaMemcpyDefault is only allowed on systems that support unified virtual
addressing.


CUDA Runtime API vRelease Version  |  409


Modules











See also:

cudaMemcpyFromSymbol, cudaGraphMemcpyNodeSetParams,
cudaGraphMemcpyNodeSetParamsToSymbol, cudaGraphAddMemcpyNode,
cudaGraphMemcpyNodeGetParams
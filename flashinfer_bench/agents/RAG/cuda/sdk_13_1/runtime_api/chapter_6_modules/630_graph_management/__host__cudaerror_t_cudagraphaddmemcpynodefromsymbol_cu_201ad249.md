# __host__cudaError_t cudaGraphAddMemcpyNodeFromSymbol (cudaGraphNode_t *pGraphNode, cudaGraph_t graph, const cudaGraphNode_t *pDependencies, size_t numDependencies, void *dst, const void *symbol, size_t count, size_t offset, cudaMemcpyKind kind)

Creates a memcpy node to copy from a symbol on the device and adds it to a graph.

##### Parameters

**pGraphNode**

  - Returns newly created node
**graph**

  - Graph to which to add the node
**pDependencies**

  - Dependencies of the node
**numDependencies**

  - Number of dependencies
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

Creates a new memcpy node to copy from symbol and adds it to graph with numDependencies
dependencies specified via pDependencies. It is possible for numDependencies to be 0, in


CUDA Runtime API vRelease Version  |  340


Modules


which case the node will be placed at the root of the graph. pDependencies may not have any
duplicate entries. A handle to the new node will be returned in pGraphNode.

When the graph is launched, the node will copy count bytes from the memory area pointed to by
offset bytes from the start of symbol symbol to the memory area pointed to by dst. The memory
areas may not overlap. symbol is a variable that resides in global or constant memory space. kind
can be either cudaMemcpyDeviceToHost, cudaMemcpyDeviceToDevice, or cudaMemcpyDefault.
Passing cudaMemcpyDefault is recommended, in which case the type of transfer is inferred from the
pointer values. However, cudaMemcpyDefault is only allowed on systems that support unified virtual
addressing.

Memcpy nodes have some additional restrictions with regards to managed memory, if
the system contains at least one device which has a zero value for the device attribute
cudaDevAttrConcurrentManagedAccess.











See also:

cudaMemcpyFromSymbol, cudaGraphAddMemcpyNode, cudaGraphAddMemcpyNodeToSymbol,
cudaGraphMemcpyNodeGetParams, cudaGraphMemcpyNodeSetParams,
cudaGraphMemcpyNodeSetParamsFromSymbol, cudaGraphMemcpyNodeSetParamsToSymbol,
cudaGraphCreate, cudaGraphDestroyNode, cudaGraphAddChildGraphNode,
cudaGraphAddEmptyNode, cudaGraphAddKernelNode, cudaGraphAddHostNode,
cudaGraphAddMemsetNode


CUDA Runtime API vRelease Version  |  341


Modules
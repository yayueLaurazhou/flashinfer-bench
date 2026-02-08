# __host__cudaError_t cudaGraphAddMemcpyNodeToSymbol (cudaGraphNode_t *pGraphNode, cudaGraph_t graph, const cudaGraphNode_t *pDependencies, size_t numDependencies, const void *symbol, const void *src, size_t count, size_t offset, cudaMemcpyKind kind)

Creates a memcpy node to copy to a symbol on the device and adds it to a graph.

##### Parameters

**pGraphNode**

  - Returns newly created node
**graph**

  - Graph to which to add the node
**pDependencies**

  - Dependencies of the node
**numDependencies**

  - Number of dependencies
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

Creates a new memcpy node to copy to symbol and adds it to graph with numDependencies
dependencies specified via pDependencies. It is possible for numDependencies to be 0, in
which case the node will be placed at the root of the graph. pDependencies may not have any
duplicate entries. A handle to the new node will be returned in pGraphNode.

When the graph is launched, the node will copy count bytes from the memory area pointed to by
src to the memory area pointed to by offset bytes from the start of symbol symbol. The memory


CUDA Runtime API vRelease Version  |  342


Modules


areas may not overlap. symbol is a variable that resides in global or constant memory space. kind
can be either cudaMemcpyHostToDevice, cudaMemcpyDeviceToDevice, or cudaMemcpyDefault.
Passing cudaMemcpyDefault is recommended, in which case the type of transfer is inferred from the
pointer values. However, cudaMemcpyDefault is only allowed on systems that support unified virtual
addressing.

Memcpy nodes have some additional restrictions with regards to managed memory, if
the system contains at least one device which has a zero value for the device attribute
cudaDevAttrConcurrentManagedAccess.











See also:

cudaMemcpyToSymbol, cudaGraphAddMemcpyNode, cudaGraphAddMemcpyNodeFromSymbol,
cudaGraphMemcpyNodeGetParams, cudaGraphMemcpyNodeSetParams,
cudaGraphMemcpyNodeSetParamsToSymbol, cudaGraphMemcpyNodeSetParamsFromSymbol,
cudaGraphCreate, cudaGraphDestroyNode, cudaGraphAddChildGraphNode,
cudaGraphAddEmptyNode, cudaGraphAddKernelNode, cudaGraphAddHostNode,
cudaGraphAddMemsetNode
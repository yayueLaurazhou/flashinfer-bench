# __host__cudaError_t cudaGraphAddMemcpyNode1D (cudaGraphNode_t *pGraphNode, cudaGraph_t graph, const cudaGraphNode_t *pDependencies, size_t numDependencies, void *dst, const void *src, size_t count, cudaMemcpyKind kind)

Creates a 1D memcpy node and adds it to a graph.

##### Parameters

**pGraphNode**

  - Returns newly created node
**graph**

  - Graph to which to add the node
**pDependencies**

  - Dependencies of the node
**numDependencies**

  - Number of dependencies


CUDA Runtime API vRelease Version  |  338


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

Creates a new 1D memcpy node and adds it to graph with numDependencies dependencies
specified via pDependencies. It is possible for numDependencies to be 0, in which case the
node will be placed at the root of the graph. pDependencies may not have any duplicate entries. A
handle to the new node will be returned in pGraphNode.

When the graph is launched, the node will copy count bytes from the memory area pointed to by
src to the memory area pointed to by dst, where kind specifies the direction of the copy, and
must be one of cudaMemcpyHostToHost, cudaMemcpyHostToDevice, cudaMemcpyDeviceToHost,
cudaMemcpyDeviceToDevice, or cudaMemcpyDefault. Passing cudaMemcpyDefault is
recommended, in which case the type of transfer is inferred from the pointer values. However,
cudaMemcpyDefault is only allowed on systems that support unified virtual addressing. Launching
a memcpy node with dst and src pointers that do not match the direction of the copy results in an
undefined behavior.

Memcpy nodes have some additional restrictions with regards to managed memory, if
the system contains at least one device which has a zero value for the device attribute
cudaDevAttrConcurrentManagedAccess.











See also:


CUDA Runtime API vRelease Version  |  339


Modules


cudaMemcpy, cudaGraphAddMemcpyNode, cudaGraphMemcpyNodeGetParams,
cudaGraphMemcpyNodeSetParams, cudaGraphMemcpyNodeSetParams1D, cudaGraphCreate,
cudaGraphDestroyNode, cudaGraphAddChildGraphNode, cudaGraphAddEmptyNode,
cudaGraphAddKernelNode, cudaGraphAddHostNode, cudaGraphAddMemsetNode
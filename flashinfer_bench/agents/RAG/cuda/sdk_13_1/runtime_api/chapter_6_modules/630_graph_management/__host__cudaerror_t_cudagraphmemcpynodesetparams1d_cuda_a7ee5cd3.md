# __host__cudaError_t cudaGraphMemcpyNodeSetParams1D (cudaGraphNode_t node, void *dst, const void *src, size_t count, cudaMemcpyKind kind)

Sets a memcpy node's parameters to perform a 1-dimensional copy.

##### Parameters

**node**

  - Node to set the parameters for
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

Sets the parameters of memcpy node node to the copy described by the provided parameters.

When the graph is launched, the node will copy count bytes from the memory area pointed to by
src to the memory area pointed to by dst, where kind specifies the direction of the copy, and
must be one of cudaMemcpyHostToHost, cudaMemcpyHostToDevice, cudaMemcpyDeviceToHost,
cudaMemcpyDeviceToDevice, or cudaMemcpyDefault. Passing cudaMemcpyDefault is
recommended, in which case the type of transfer is inferred from the pointer values. However,
cudaMemcpyDefault is only allowed on systems that support unified virtual addressing. Launching
a memcpy node with dst and src pointers that do not match the direction of the copy results in an
undefined behavior.











CUDA Runtime API vRelease Version  |  408


Modules


See also:

cudaMemcpy, cudaGraphMemcpyNodeSetParams, cudaGraphAddMemcpyNode,
cudaGraphMemcpyNodeGetParams
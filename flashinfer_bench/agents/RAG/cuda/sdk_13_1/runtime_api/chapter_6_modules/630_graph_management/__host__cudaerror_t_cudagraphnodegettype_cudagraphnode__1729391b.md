# __host__cudaError_t cudaGraphNodeGetType (cudaGraphNode_t node, cudaGraphNodeType *pType)

Returns a node's type.

##### Parameters

**node**

  - Node to query


CUDA Runtime API vRelease Version  |  419


Modules



**pType**

  - Pointer to return the node type

##### Returns

cudaSuccess, cudaErrorInvalidValue

##### Description

Returns the node type of node in pType.











See also:

cudaGraphGetNodes, cudaGraphGetRootNodes, cudaGraphChildGraphNodeGetGraph,
cudaGraphKernelNodeGetParams, cudaGraphKernelNodeSetParams, cudaGraphHostNodeGetParams,
cudaGraphHostNodeSetParams, cudaGraphMemcpyNodeGetParams,
cudaGraphMemcpyNodeSetParams, cudaGraphMemsetNodeGetParams,
cudaGraphMemsetNodeSetParams
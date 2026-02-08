# __host__cudaError_t cudaGraphNodeSetParams (cudaGraphNode_t node, cudaGraphNodeParams *nodeParams)

Update's a graph node's parameters.

##### Parameters

**node**

  - Node to set the parameters for


CUDA Runtime API vRelease Version  |  421


Modules


**nodeParams**

  - Parameters to copy

##### Returns

cudaSuccess, cudaErrorInvalidValue, cudaErrorInvalidDeviceFunction, cudaErrorNotSupported

##### Description

Sets the parameters of graph node node to nodeParams. The node type specified by
nodeParams->type must match the type of node. nodeParams must be fully initialized and all
unused bytes (reserved, padding) zeroed.

Modifying parameters is not supported for node types cudaGraphNodeTypeMemAlloc and
cudaGraphNodeTypeMemFree.











See also:

cudaGraphAddNode, cudaGraphExecNodeSetParams
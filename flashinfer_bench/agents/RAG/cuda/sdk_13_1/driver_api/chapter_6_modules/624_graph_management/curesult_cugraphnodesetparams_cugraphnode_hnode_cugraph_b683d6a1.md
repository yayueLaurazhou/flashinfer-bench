# CUresult cuGraphNodeSetParams (CUgraphNode hNode, CUgraphNodeParams *nodeParams)

Update's a graph node's parameters.

###### Parameters

**hNode**

  - Node to set the parameters for
**nodeParams**

  - Parameters to copy

###### Returns

CUDA_SUCCESS, CUDA_ERROR_INVALID_VALUE, CUDA_ERROR_NOT_SUPPORTED

###### Description

Sets the parameters of graph node hNode to nodeParams. The node type specified by
nodeParams->type must match the type of hNode. nodeParams must be fully initialized and all
unused bytes (reserved, padding) zeroed.

Modifying parameters is not supported for node types CU_GRAPH_NODE_TYPE_MEM_ALLOC
and CU_GRAPH_NODE_TYPE_MEM_FREE.





See also:

cuGraphAddNode, cuGraphExecNodeSetParams
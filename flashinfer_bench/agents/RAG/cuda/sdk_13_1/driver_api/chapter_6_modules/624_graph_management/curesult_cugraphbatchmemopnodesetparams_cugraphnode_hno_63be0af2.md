# CUresult cuGraphBatchMemOpNodeSetParams (CUgraphNode hNode, const CUDA_BATCH_MEM_OP_NODE_PARAMS *nodeParams)

Sets a batch mem op node's parameters.

###### Parameters

**hNode**

  - Node to set the parameters for
**nodeParams**

  - Parameters to copy

###### Returns

CUDA_SUCCESS, CUDA_ERROR_INVALID_VALUE, CUDA_ERROR_INVALID_HANDLE,
CUDA_ERROR_OUT_OF_MEMORY

###### Description

Sets the parameters of batch mem op node hNode to nodeParams.

The paramArray inside nodeParams is copied and therefore it can be freed after the call returns.





See also:

cuGraphNodeSetParams, cuStreamBatchMemOp, cuGraphAddBatchMemOpNode,
cuGraphBatchMemOpNodeGetParams


CUDA Driver API TRM-06703-001 _vRelease Version  |  436


Modules
# CUresult cuGraphBatchMemOpNodeGetParams (CUgraphNode hNode, CUDA_BATCH_MEM_OP_NODE_PARAMS *nodeParams_out)

Returns a batch mem op node's parameters.

###### Parameters

**hNode**

  - Node to get the parameters for
**nodeParams_out**

  - Pointer to return the parameters

###### Returns

CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
CUDA_ERROR_INVALID_VALUE

###### Description

Returns the parameters of batch mem op node hNode in nodeParams_out. The paramArray
returned in nodeParams_out is owned by the node. This memory remains valid until the
node is destroyed or its parameters are modified, and should not be modified directly. Use
cuGraphBatchMemOpNodeSetParams to update the parameters of this node.


Note:


CUDA Driver API TRM-06703-001 _vRelease Version  |  435


Modules







See also:

cuStreamBatchMemOp, cuGraphAddBatchMemOpNode, cuGraphBatchMemOpNodeSetParams
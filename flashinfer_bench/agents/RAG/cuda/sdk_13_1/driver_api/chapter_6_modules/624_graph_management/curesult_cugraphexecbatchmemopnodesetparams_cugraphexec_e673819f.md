# CUresult cuGraphExecBatchMemOpNodeSetParams (CUgraphExec hGraphExec, CUgraphNode hNode, const CUDA_BATCH_MEM_OP_NODE_PARAMS *nodeParams)

Sets the parameters for a batch mem op node in the given graphExec.

###### Parameters

**hGraphExec**

  - The executable graph in which to set the specified node
**hNode**

  - Batch mem op node from the graph from which graphExec was instantiated
**nodeParams**

  - Updated Parameters to set

###### Returns

CUDA_SUCCESS, CUDA_ERROR_INVALID_VALUE,

###### Description

Sets the parameters of a batch mem op node in an executable graph hGraphExec. The node is
identified by the corresponding node hNode in the non-executable graph, from which the executable
graph was instantiated.

The following fields on operations may be modified on an executable graph:

op.waitValue.address op.waitValue.value[64] op.waitValue.flags bits corresponding to wait type
(i.e. CU_STREAM_WAIT_VALUE_FLUSH bit cannot be modified) op.writeValue.address
op.writeValue.value[64]

Other fields, such as the context, count or type of operations, and other types of operations such as
membars, may not be modified.

hNode must not have been removed from the original graph.

The modifications only affect future launches of hGraphExec. Already enqueued or running
launches of hGraphExec are not affected by this call. hNode is also not modified by this call.

The paramArray inside nodeParams is copied and therefore it can be freed after the call returns.


CUDA Driver API TRM-06703-001 _vRelease Version  |  445


Modules





See also:

cuGraphExecNodeSetParams, cuStreamBatchMemOp, cuGraphAddBatchMemOpNode,
cuGraphBatchMemOpNodeGetParams, cuGraphBatchMemOpNodeSetParams, cuGraphInstantiate
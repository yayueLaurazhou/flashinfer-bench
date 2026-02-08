# CUresult cuGraphAddBatchMemOpNode (CUgraphNode *phGraphNode, CUgraph hGraph, const CUgraphNode *dependencies, size_t numDependencies, const CUDA_BATCH_MEM_OP_NODE_PARAMS *nodeParams)

Creates a batch memory operation node and adds it to a graph.

###### Parameters

**phGraphNode**

  - Returns newly created node
**hGraph**

  - Graph to which to add the node
**dependencies**

  - Dependencies of the node
**numDependencies**

  - Number of dependencies
**nodeParams**

  - Parameters for the node

###### Returns

CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
CUDA_ERROR_NOT_SUPPORTED, CUDA_ERROR_INVALID_VALUE

###### Description

Creates a new batch memory operation node and adds it to hGraph with numDependencies
dependencies specified via dependencies and arguments specified in nodeParams. It is possible
for numDependencies to be 0, in which case the node will be placed at the root of the graph.
dependencies may not have any duplicate entries. A handle to the new node will be returned in
phGraphNode.


CUDA Driver API TRM-06703-001 _vRelease Version  |  416


Modules


When the node is added, the paramArray inside nodeParams is copied and therefore it can be freed
after the call returns.


Note:


Warning: Improper use of this API may deadlock the application. Synchronization ordering established
through this API is not visible to CUDA. CUDA tasks that are (even indirectly) ordered by this API
should also have that order expressed with CUDA-visible dependencies such as events. This ensures that
the scheduler does not serialize them in an improper order.





See also:

cuGraphAddNode, cuStreamBatchMemOp, cuStreamWaitValue32, cuStreamWriteValue32,
cuStreamWaitValue64, cuStreamWriteValue64, cuGraphBatchMemOpNodeGetParams,
cuGraphBatchMemOpNodeSetParams, cuGraphCreate, cuGraphDestroyNode,
cuGraphAddChildGraphNode, cuGraphAddEmptyNode, cuGraphAddKernelNode,
cuGraphAddMemcpyNode, cuGraphAddMemsetNode
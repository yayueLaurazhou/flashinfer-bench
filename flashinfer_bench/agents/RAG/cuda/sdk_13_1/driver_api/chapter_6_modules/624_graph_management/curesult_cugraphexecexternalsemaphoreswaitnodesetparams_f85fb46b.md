# CUresult cuGraphExecExternalSemaphoresWaitNodeSetParams (CUgraphExec hGraphExec, CUgraphNode hNode, const CUDA_EXT_SEM_WAIT_NODE_PARAMS *nodeParams)

Sets the parameters for an external semaphore wait node in the given graphExec.

###### Parameters

**hGraphExec**

  - The executable graph in which to set the specified node
**hNode**

  - semaphore wait node from the graph from which graphExec was instantiated
**nodeParams**

  - Updated Parameters to set

###### Returns

CUDA_SUCCESS, CUDA_ERROR_INVALID_VALUE,

###### Description

Sets the parameters of an external semaphore wait node in an executable graph hGraphExec. The
node is identified by the corresponding node hNode in the non-executable graph, from which the
executable graph was instantiated.

hNode must not have been removed from the original graph.

The modifications only affect future launches of hGraphExec. Already enqueued or running
launches of hGraphExec are not affected by this call. hNode is also not modified by this call.

Changing nodeParams->numExtSems is not supported.





CUDA Driver API TRM-06703-001 _vRelease Version  |  451


Modules


See also:

cuGraphExecNodeSetParams, cuGraphAddExternalSemaphoresWaitNode,
cuImportExternalSemaphore, cuSignalExternalSemaphoresAsync, cuWaitExternalSemaphoresAsync,
cuGraphExecKernelNodeSetParams, cuGraphExecMemcpyNodeSetParams,
cuGraphExecMemsetNodeSetParams, cuGraphExecHostNodeSetParams,
cuGraphExecChildGraphNodeSetParams, cuGraphExecEventRecordNodeSetEvent,
cuGraphExecEventWaitNodeSetEvent, cuGraphExecExternalSemaphoresSignalNodeSetParams,
cuGraphExecUpdate, cuGraphInstantiate
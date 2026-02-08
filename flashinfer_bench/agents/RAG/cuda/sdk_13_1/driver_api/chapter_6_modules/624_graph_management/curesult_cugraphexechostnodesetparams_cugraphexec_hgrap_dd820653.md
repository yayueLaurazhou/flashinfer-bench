# CUresult cuGraphExecHostNodeSetParams (CUgraphExec hGraphExec, CUgraphNode hNode, const CUDA_HOST_NODE_PARAMS *nodeParams)

Sets the parameters for a host node in the given graphExec.

###### Parameters

**hGraphExec**

  - The executable graph in which to set the specified node
**hNode**

  - Host node from the graph which was used to instantiate graphExec
**nodeParams**

  - The updated parameters to set

###### Returns

CUDA_SUCCESS, CUDA_ERROR_INVALID_VALUE,


CUDA Driver API TRM-06703-001 _vRelease Version  |  453


Modules

###### Description

Updates the work represented by hNode in hGraphExec as though hNode had contained
nodeParams at instantiation. hNode must remain in the graph which was used to instantiate
hGraphExec. Changed edges to and from hNode are ignored.

The modifications only affect future launches of hGraphExec. Already enqueued or running
launches of hGraphExec are not affected by this call. hNode is also not modified by this call.





See also:

cuGraphExecNodeSetParams, cuGraphAddHostNode, cuGraphHostNodeSetParams,
cuGraphExecKernelNodeSetParams, cuGraphExecMemcpyNodeSetParams,
cuGraphExecMemsetNodeSetParams, cuGraphExecChildGraphNodeSetParams,
cuGraphExecEventRecordNodeSetEvent, cuGraphExecEventWaitNodeSetEvent,
cuGraphExecExternalSemaphoresSignalNodeSetParams,
cuGraphExecExternalSemaphoresWaitNodeSetParams, cuGraphExecUpdate, cuGraphInstantiate
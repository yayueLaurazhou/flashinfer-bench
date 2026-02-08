# CUresult cuGraphExecEventWaitNodeSetEvent (CUgraphExec hGraphExec, CUgraphNode hNode, CUevent event)

Sets the event for an event wait node in the given graphExec.

###### Parameters

**hGraphExec**

  - The executable graph in which to set the specified node
**hNode**

  - event wait node from the graph from which graphExec was instantiated
**event**

  - Updated event to use

###### Returns

CUDA_SUCCESS, CUDA_ERROR_INVALID_VALUE,

###### Description

Sets the event of an event wait node in an executable graph hGraphExec. The node is identified by
the corresponding node hNode in the non-executable graph, from which the executable graph was
instantiated.

The modifications only affect future launches of hGraphExec. Already enqueued or running
launches of hGraphExec are not affected by this call. hNode is also not modified by this call.





See also:

cuGraphExecNodeSetParams, cuGraphAddEventWaitNode,
cuGraphEventWaitNodeGetEvent, cuGraphEventRecordNodeSetEvent,
cuEventRecordWithFlags, cuStreamWaitEvent, cuGraphExecKernelNodeSetParams,
cuGraphExecMemcpyNodeSetParams, cuGraphExecMemsetNodeSetParams,
cuGraphExecHostNodeSetParams, cuGraphExecChildGraphNodeSetParams,
cuGraphExecEventRecordNodeSetEvent, cuGraphExecExternalSemaphoresSignalNodeSetParams,
cuGraphExecExternalSemaphoresWaitNodeSetParams, cuGraphExecUpdate, cuGraphInstantiate


CUDA Driver API TRM-06703-001 _vRelease Version  |  449


Modules
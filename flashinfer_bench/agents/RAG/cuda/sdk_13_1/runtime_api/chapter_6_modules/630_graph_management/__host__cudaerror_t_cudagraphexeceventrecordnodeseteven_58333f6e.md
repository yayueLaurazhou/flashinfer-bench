# __host__cudaError_t cudaGraphExecEventRecordNodeSetEvent (cudaGraphExec_t hGraphExec, cudaGraphNode_t hNode, cudaEvent_t event)

Sets the event for an event record node in the given graphExec.

##### Parameters

**hGraphExec**

  - The executable graph in which to set the specified node
**hNode**

  - Event record node from the graph from which graphExec was instantiated
**event**

  - Updated event to use

##### Returns

cudaSuccess, cudaErrorInvalidValue,

##### Description

Sets the event of an event record node in an executable graph hGraphExec. The node is identified
by the corresponding node hNode in the non-executable graph, from which the executable graph was
instantiated.

The modifications only affect future launches of hGraphExec. Already enqueued or running
launches of hGraphExec are not affected by this call. hNode is also not modified by this call.











CUDA Runtime API vRelease Version  |  359


Modules


See also:

cudaGraphExecNodeSetParams, cudaGraphAddEventRecordNode,
cudaGraphEventRecordNodeGetEvent, cudaGraphEventWaitNodeSetEvent,
cudaEventRecordWithFlags, cudaStreamWaitEvent, cudaGraphExecKernelNodeSetParams,
cudaGraphExecMemcpyNodeSetParams, cudaGraphExecMemsetNodeSetParams,
cudaGraphExecHostNodeSetParams, cudaGraphExecChildGraphNodeSetParams,
cudaGraphExecEventWaitNodeSetEvent, cudaGraphExecExternalSemaphoresSignalNodeSetParams,
cudaGraphExecExternalSemaphoresWaitNodeSetParams, cudaGraphExecUpdate,
cudaGraphInstantiate
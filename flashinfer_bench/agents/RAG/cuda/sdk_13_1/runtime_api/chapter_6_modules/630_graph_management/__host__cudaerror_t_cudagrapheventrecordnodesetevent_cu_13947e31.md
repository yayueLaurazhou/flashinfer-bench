# __host__cudaError_t cudaGraphEventRecordNodeSetEvent (cudaGraphNode_t node, cudaEvent_t event)

Sets an event record node's event.

##### Parameters

**node**
**event**

  - Event to use

##### Returns

cudaSuccess, cudaErrorInvalidValue

##### Description

Sets the event of event record node hNode to event.











See also:

cudaGraphNodeSetParams, cudaGraphAddEventRecordNode, cudaGraphEventRecordNodeGetEvent,
cudaGraphEventWaitNodeSetEvent, cudaEventRecordWithFlags, cudaStreamWaitEvent
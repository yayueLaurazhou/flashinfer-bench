# __host__cudaError_t cudaGraphEventRecordNodeGetEvent (cudaGraphNode_t node, cudaEvent_t *event_out)

Returns the event associated with an event record node.

##### Parameters

**node**
**event_out**

  - Pointer to return the event

##### Returns

cudaSuccess, cudaErrorInvalidValue

##### Description

Returns the event of event record node hNode in event_out.











See also:

cudaGraphAddEventRecordNode, cudaGraphEventRecordNodeSetEvent,
cudaGraphEventWaitNodeGetEvent, cudaEventRecordWithFlags, cudaStreamWaitEvent


CUDA Runtime API vRelease Version  |  354


Modules
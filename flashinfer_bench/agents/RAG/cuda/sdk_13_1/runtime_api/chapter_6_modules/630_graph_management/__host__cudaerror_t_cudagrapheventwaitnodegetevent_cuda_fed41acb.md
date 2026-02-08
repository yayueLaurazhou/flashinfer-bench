# __host__cudaError_t cudaGraphEventWaitNodeGetEvent (cudaGraphNode_t node, cudaEvent_t *event_out)

Returns the event associated with an event wait node.

##### Parameters

**node**
**event_out**

  - Pointer to return the event


CUDA Runtime API vRelease Version  |  355


Modules

##### Returns

cudaSuccess, cudaErrorInvalidValue

##### Description

Returns the event of event wait node hNode in event_out.











See also:

cudaGraphAddEventWaitNode, cudaGraphEventWaitNodeSetEvent,
cudaGraphEventRecordNodeGetEvent, cudaEventRecordWithFlags, cudaStreamWaitEvent
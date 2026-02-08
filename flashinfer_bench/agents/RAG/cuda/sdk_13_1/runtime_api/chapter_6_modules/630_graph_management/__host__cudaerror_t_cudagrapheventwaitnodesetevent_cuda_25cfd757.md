# __host__cudaError_t cudaGraphEventWaitNodeSetEvent (cudaGraphNode_t node, cudaEvent_t event)

Sets an event wait node's event.

##### Parameters

**node**
**event**

  - Event to use

##### Returns

cudaSuccess, cudaErrorInvalidValue

##### Description

Sets the event of event wait node hNode to event.





CUDA Runtime API vRelease Version  |  356


Modules







See also:

cudaGraphNodeSetParams, cudaGraphAddEventWaitNode, cudaGraphEventWaitNodeGetEvent,
cudaGraphEventRecordNodeSetEvent, cudaEventRecordWithFlags, cudaStreamWaitEvent
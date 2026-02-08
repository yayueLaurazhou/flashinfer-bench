# CUresult cuGraphEventRecordNodeSetEvent (CUgraphNode hNode, CUevent event)

Sets an event record node's event.

###### Parameters

**hNode**

  - Node to set the event for
**event**

  - Event to use

###### Returns

CUDA_SUCCESS, CUDA_ERROR_INVALID_VALUE, CUDA_ERROR_INVALID_HANDLE,
CUDA_ERROR_OUT_OF_MEMORY

###### Description

Sets the event of event record node hNode to event.





See also:

cuGraphNodeSetParams, cuGraphAddEventRecordNode, cuGraphEventRecordNodeGetEvent,
cuGraphEventWaitNodeSetEvent, cuEventRecordWithFlags, cuStreamWaitEvent
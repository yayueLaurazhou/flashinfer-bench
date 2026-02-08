# CUresult cuGraphEventWaitNodeSetEvent (CUgraphNode hNode, CUevent event)

Sets an event wait node's event.

###### Parameters

**hNode**

  - Node to set the event for
**event**

  - Event to use

###### Returns

CUDA_SUCCESS, CUDA_ERROR_INVALID_VALUE, CUDA_ERROR_INVALID_HANDLE,
CUDA_ERROR_OUT_OF_MEMORY

###### Description

Sets the event of event wait node hNode to event.





CUDA Driver API TRM-06703-001 _vRelease Version  |  444


Modules


See also:

cuGraphNodeSetParams, cuGraphAddEventWaitNode, cuGraphEventWaitNodeGetEvent,
cuGraphEventRecordNodeSetEvent, cuEventRecordWithFlags, cuStreamWaitEvent
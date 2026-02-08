# CUresult cuGraphEventRecordNodeGetEvent (CUgraphNode hNode, CUevent *event_out)

Returns the event associated with an event record node.

###### Parameters

**hNode**

  - Node to get the event for
**event_out**

  - Pointer to return the event

###### Returns

CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
CUDA_ERROR_INVALID_VALUE

###### Description

Returns the event of event record node hNode in event_out.





See also:

cuGraphAddEventRecordNode, cuGraphEventRecordNodeSetEvent,
cuGraphEventWaitNodeGetEvent, cuEventRecordWithFlags, cuStreamWaitEvent


CUDA Driver API TRM-06703-001 _vRelease Version  |  442


Modules
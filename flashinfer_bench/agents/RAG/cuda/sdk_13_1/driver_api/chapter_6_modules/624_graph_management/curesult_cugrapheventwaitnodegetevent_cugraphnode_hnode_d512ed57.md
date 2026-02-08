# CUresult cuGraphEventWaitNodeGetEvent (CUgraphNode hNode, CUevent *event_out)

Returns the event associated with an event wait node.

###### Parameters

**hNode**

  - Node to get the event for
**event_out**

  - Pointer to return the event


CUDA Driver API TRM-06703-001 _vRelease Version  |  443


Modules

###### Returns

CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
CUDA_ERROR_INVALID_VALUE

###### Description

Returns the event of event wait node hNode in event_out.





See also:

cuGraphAddEventWaitNode, cuGraphEventWaitNodeSetEvent, cuGraphEventRecordNodeGetEvent,
cuEventRecordWithFlags, cuStreamWaitEvent
# enum CUevent_record_flags

Event record flags

###### Values

**CU_EVENT_RECORD_DEFAULT = 0x0**


CUDA Driver API TRM-06703-001 _vRelease Version  |  40


Modules


Default event record flag
**CU_EVENT_RECORD_EXTERNAL = 0x1**
When using stream capture, create an event record node instead of the default behavior. This flag is
invalid when used outside of capture.
# enum CUevent_flags

Event creation flags

###### Values

**CU_EVENT_DEFAULT = 0x0**
Default event flag
**CU_EVENT_BLOCKING_SYNC = 0x1**
Event uses blocking synchronization
**CU_EVENT_DISABLE_TIMING = 0x2**
Event will not record timing data
**CU_EVENT_INTERPROCESS = 0x4**
Event is suitable for interprocess use. CU_EVENT_DISABLE_TIMING must be set
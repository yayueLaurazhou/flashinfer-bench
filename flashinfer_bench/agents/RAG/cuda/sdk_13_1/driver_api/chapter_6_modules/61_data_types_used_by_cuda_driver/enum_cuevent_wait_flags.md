# enum CUevent_wait_flags

Event wait flags

###### Values

**CU_EVENT_WAIT_DEFAULT = 0x0**
Default event wait flag
**CU_EVENT_WAIT_EXTERNAL = 0x1**
When using stream capture, create an event wait node instead of the default behavior. This flag is
invalid when used outside of capture.
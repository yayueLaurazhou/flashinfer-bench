# CUlaunchAttributeValue::@7 CUlaunchAttributeValue::programmaticEvent

Value of launch attribute CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_EVENT with the
following fields:

event - Event to fire when all blocks trigger it.

###### ‣ CUevent

record flags, see cuEventRecordWithFlags. Does not

###### ‣ Event

accept :CU_EVENT_RECORD_EXTERNAL.
                 - If this is set to non-0, each block launch will automatically trigger

###### ‣ triggerAtBlockStart

the event.


CUDA Driver API TRM-06703-001 _vRelease Version  |  725


Data Structures
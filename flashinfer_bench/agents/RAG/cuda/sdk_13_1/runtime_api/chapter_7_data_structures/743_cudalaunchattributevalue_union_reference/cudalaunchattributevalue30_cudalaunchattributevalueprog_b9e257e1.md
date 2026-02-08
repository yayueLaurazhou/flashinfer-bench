# cudaLaunchAttributeValue::@30 cudaLaunchAttributeValue::programmaticEvent

Value of launch attribute cudaLaunchAttributeProgrammaticEvent with the following fields:

event - Event to fire when all blocks trigger it.

##### ‣ cudaEvent_t

flags; - Event record flags, see cudaEventRecordWithFlags. Does not accept

##### ‣ int

cudaEventRecordExternal.
triggerAtBlockStart - If this is set to non-0, each block launch will automatically trigger the

##### ‣ int

event.
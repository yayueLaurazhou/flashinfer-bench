# enum CUstreamWaitValue_flags

Flags for cuStreamWaitValue32 and cuStreamWaitValue64

###### Values

**CU_STREAM_WAIT_VALUE_GEQ = 0x0**
Wait until (int32_t)(*addr - value) >= 0 (or int64_t for 64 bit values). Note this is a cyclic
comparison which ignores wraparound. (Default behavior.)
**CU_STREAM_WAIT_VALUE_EQ = 0x1**
Wait until *addr == value.
**CU_STREAM_WAIT_VALUE_AND = 0x2**
Wait until (*addr & value) != 0.
**CU_STREAM_WAIT_VALUE_NOR = 0x3**
Wait until ~(*addr | value) != 0. Support for this operation can be queried with
cuDeviceGetAttribute() and
CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_WAIT_VALUE_NOR.
**CU_STREAM_WAIT_VALUE_FLUSH = 1<<30**
Follow the wait operation with a flush of outstanding remote writes. This means that, if a remote
write operation is guaranteed to have reached the device before the wait can be satisfied, that
write is guaranteed to be visible to downstream device work. The device is permitted to reorder
remote writes internally. For example, this flag would be required if two remote writes arrive in
a defined order, the wait is satisfied by the second write, and downstream work needs to observe
the first write. Support for this operation is restricted to selected platforms and can be queried with
CU_DEVICE_ATTRIBUTE_CAN_FLUSH_REMOTE_WRITES.
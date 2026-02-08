# 6.21. Stream Memory Operations

This section describes the stream memory operations of the low-level CUDA driver application
programming interface.

Support for the CU_STREAM_WAIT_VALUE_NOR flag can be queried with
CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_WAIT_VALUE_NOR_V2.

Support for the cuStreamWriteValue64() and cuStreamWaitValue64() functions, as well as for the
CU_STREAM_MEM_OP_WAIT_VALUE_64 and CU_STREAM_MEM_OP_WRITE_VALUE_64
flags, can be queried with CU_DEVICE_ATTRIBUTE_CAN_USE_64_BIT_STREAM_MEM_OPS.

Support for both CU_STREAM_WAIT_VALUE_FLUSH and
CU_STREAM_MEM_OP_FLUSH_REMOTE_WRITES requires dedicated
platform hardware features and can be queried with cuDeviceGetAttribute() and
CU_DEVICE_ATTRIBUTE_CAN_FLUSH_REMOTE_WRITES.

Note that all memory pointers passed as parameters to these operations are device pointers. Where
necessary a device pointer should be obtained, for example with cuMemHostGetDevicePointer().

None of the operations accepts pointers to managed memory buffers (cuMemAllocManaged).


Note:


Warning: Improper use of these APIs may deadlock the application. Synchronization ordering
established through these APIs is not visible to CUDA. CUDA tasks that are (even indirectly) ordered by
these APIs should also have that order expressed with CUDA-visible dependencies such as events. This
ensures that the scheduler does not serialize them in an improper order.
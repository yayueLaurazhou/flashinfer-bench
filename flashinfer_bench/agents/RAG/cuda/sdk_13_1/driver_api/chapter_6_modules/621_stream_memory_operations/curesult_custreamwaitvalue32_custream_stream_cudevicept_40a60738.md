# CUresult cuStreamWaitValue32 (CUstream stream, CUdeviceptr addr, cuuint32_t value, unsigned int flags)

Wait on a memory location.

###### Parameters

**stream**
The stream to synchronize on the memory location.
**addr**
The memory location to wait on.
**value**
The value to compare with the memory location.
**flags**
See CUstreamWaitValue_flags.

###### Returns

CUDA_SUCCESS, CUDA_ERROR_INVALID_VALUE, CUDA_ERROR_NOT_SUPPORTED

###### Description

Enqueues a synchronization of the stream on the given memory location. Work ordered after the
operation will block until the given condition on the memory is satisfied. By default, the condition is to
wait for (int32_t)(*addr - value) >= 0, a cyclic greater-or-equal. Other condition types can be specified
via flags.

If the memory was registered via cuMemHostRegister(), the device pointer should be obtained
with cuMemHostGetDevicePointer(). This function cannot be used with managed memory
(cuMemAllocManaged).

Support for CU_STREAM_WAIT_VALUE_NOR can be queried with cuDeviceGetAttribute() and
CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_WAIT_VALUE_NOR_V2.


Note:


Warning: Improper use of this API may deadlock the application. Synchronization ordering established
through this API is not visible to CUDA. CUDA tasks that are (even indirectly) ordered by this API
should also have that order expressed with CUDA-visible dependencies such as events. This ensures that
the scheduler does not serialize them in an improper order.


Note:


Note that this function may also return error codes from previous, asynchronous launches.


CUDA Driver API TRM-06703-001 _vRelease Version  |  378


Modules


See also:

cuStreamWaitValue64, cuStreamWriteValue32, cuStreamWriteValue64, cuStreamBatchMemOp,
cuMemHostRegister, cuStreamWaitEvent
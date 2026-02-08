# CUresult cuStreamWriteValue64 (CUstream stream, CUdeviceptr addr, cuuint64_t value, unsigned int flags)

Write a value to memory.

###### Parameters

**stream**
The stream to do the write in.
**addr**
The device address to write to.
**value**
The value to write.
**flags**
See CUstreamWriteValue_flags.

###### Returns

CUDA_SUCCESS, CUDA_ERROR_INVALID_VALUE, CUDA_ERROR_NOT_SUPPORTED

###### Description

Write a value to memory.

If the memory was registered via cuMemHostRegister(), the device pointer should be obtained with
cuMemHostGetDevicePointer().

Support for this can be queried with cuDeviceGetAttribute() and
CU_DEVICE_ATTRIBUTE_CAN_USE_64_BIT_STREAM_MEM_OPS.


Note:


Note that this function may also return error codes from previous, asynchronous launches.


See also:

cuStreamWriteValue32, cuStreamWaitValue32, cuStreamWaitValue64, cuStreamBatchMemOp,
cuMemHostRegister, cuEventRecord
# CUresult cuStreamWriteValue32 (CUstream stream, CUdeviceptr addr, cuuint32_t value, unsigned int flags)

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

If the memory was registered via cuMemHostRegister(), the device pointer should be obtained
with cuMemHostGetDevicePointer(). This function cannot be used with managed memory
(cuMemAllocManaged).


Note:


Note that this function may also return error codes from previous, asynchronous launches.


See also:

cuStreamWriteValue64, cuStreamWaitValue32, cuStreamWaitValue64, cuStreamBatchMemOp,
cuMemHostRegister, cuEventRecord


CUDA Driver API TRM-06703-001 _vRelease Version  |  380


Modules
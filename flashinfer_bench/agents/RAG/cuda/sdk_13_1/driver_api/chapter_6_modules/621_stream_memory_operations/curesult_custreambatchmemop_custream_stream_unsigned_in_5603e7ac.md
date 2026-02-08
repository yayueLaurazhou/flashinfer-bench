# CUresult cuStreamBatchMemOp (CUstream stream, unsigned int count, CUstreamBatchMemOpParams *paramArray, unsigned int flags)

Batch operations to synchronize the stream via memory operations.

###### Parameters

**stream**
The stream to enqueue the operations in.
**count**
The number of operations in the array. Must be less than 256.


CUDA Driver API TRM-06703-001 _vRelease Version  |  376


Modules


**paramArray**
The types and parameters of the individual operations.
**flags**
Reserved for future expansion; must be 0.

###### Returns

CUDA_SUCCESS, CUDA_ERROR_INVALID_VALUE, CUDA_ERROR_NOT_SUPPORTED

###### Description

This is a batch version of cuStreamWaitValue32() and cuStreamWriteValue32(). Batching operations
may avoid some performance overhead in both the API call and the device execution versus adding
them to the stream in separate API calls. The operations are enqueued in the order they appear in the
array.

See CUstreamBatchMemOpType for the full set of supported operations, and cuStreamWaitValue32(),
cuStreamWaitValue64(), cuStreamWriteValue32(), and cuStreamWriteValue64() for details of specific
operations.

See related APIs for details on querying support for specific operations.


Note:


Warning: Improper use of this API may deadlock the application. Synchronization ordering established
through this API is not visible to CUDA. CUDA tasks that are (even indirectly) ordered by this API
should also have that order expressed with CUDA-visible dependencies such as events. This ensures that
the scheduler does not serialize them in an improper order.


Note:


Note that this function may also return error codes from previous, asynchronous launches.


See also:

cuStreamWaitValue32, cuStreamWaitValue64, cuStreamWriteValue32, cuStreamWriteValue64,
cuMemHostRegister


CUDA Driver API TRM-06703-001 _vRelease Version  |  377


Modules
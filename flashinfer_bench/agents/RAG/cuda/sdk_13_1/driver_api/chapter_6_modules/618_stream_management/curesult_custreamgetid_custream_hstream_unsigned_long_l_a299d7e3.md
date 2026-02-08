# CUresult cuStreamGetId (CUstream hStream, unsigned long long *streamId)

Returns the unique Id associated with the stream handle supplied.

###### Parameters

**hStream**

  - Handle to the stream to be queried
**streamId**

  - Pointer to store the Id of the stream

###### Returns

CUDA_SUCCESS, CUDA_ERROR_INVALID_VALUE, CUDA_ERROR_INVALID_HANDLE

###### Description

Returns in streamId the unique Id which is associated with the given stream handle. The Id is unique
for the life of the program.

The stream handle hStream can refer to any of the following:

a stream created via any of the CUDA driver APIs such as cuStreamCreate and

###### **‣**

cuStreamCreateWithPriority, or their runtime API equivalents such as cudaStreamCreate,
cudaStreamCreateWithFlags and cudaStreamCreateWithPriority. Passing an invalid handle will
result in undefined behavior.
any of the special streams such as the NULL stream, CU_STREAM_LEGACY and

###### **‣**

CU_STREAM_PER_THREAD. The runtime API equivalents of these are also accepted, which are
NULL, cudaStreamLegacy and cudaStreamPerThread respectively.


Note:


Note that this function may also return error codes from previous, asynchronous launches.


See also:

cuStreamDestroy, cuStreamCreate, cuStreamGetPriority, cudaStreamGetId


CUDA Driver API TRM-06703-001 _vRelease Version  |  347


Modules
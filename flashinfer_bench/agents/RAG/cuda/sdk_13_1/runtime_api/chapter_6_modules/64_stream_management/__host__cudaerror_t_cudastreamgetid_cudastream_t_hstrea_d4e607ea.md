# __host__cudaError_t cudaStreamGetId (cudaStream_t hStream, unsigned long long *streamId)

Query the Id of a stream.

##### Parameters

**hStream**

  - Handle to the stream to be queried
**streamId**

  - Pointer to an unsigned long long in which the stream Id is returned

##### Returns

cudaSuccess, cudaErrorInvalidValue, cudaErrorInvalidResourceHandle

##### Description

Query the Id of a stream. The Id is returned in streamId. The Id is unique for the life of the program.

The stream handle hStream can refer to any of the following:


CUDA Runtime API vRelease Version  |  64


Modules


a stream created via any of the CUDA runtime APIs such as cudaStreamCreate,

##### **‣**

cudaStreamCreateWithFlags and cudaStreamCreateWithPriority, or their driver API equivalents
such as cuStreamCreate or cuStreamCreateWithPriority. Passing an invalid handle will result in
undefined behavior.
any of the special streams such as the NULL stream, cudaStreamLegacy and cudaStreamPerThread

##### **‣**

respectively. The driver API equivalents of these are also accepted which are NULL,
CU_STREAM_LEGACY and CU_STREAM_PER_THREAD.











See also:

cudaStreamCreateWithPriority, cudaStreamCreateWithFlags, cudaStreamGetPriority,
cudaStreamGetFlags, cuStreamGetId
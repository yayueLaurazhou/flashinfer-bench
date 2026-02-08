# CUresult cuStreamCreate (CUstream *phStream, unsigned int Flags)

Create a stream.

###### Parameters

**phStream**

  - Returned newly created stream
**Flags**

  - Parameters for stream creation

###### Returns

CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE,
CUDA_ERROR_OUT_OF_MEMORY


CUDA Driver API TRM-06703-001 _vRelease Version  |  337


Modules

###### Description

Creates a stream and returns a handle in phStream. The Flags argument determines behaviors of
the stream.

Valid values for Flags are:

CU_STREAM_DEFAULT: Default stream creation flag.

###### **‣**

CU_STREAM_NON_BLOCKING: Specifies that work running in the created stream may run

###### **‣**

concurrently with work in stream 0 (the NULL stream), and that the created stream should perform
no implicit synchronization with stream 0.


Note:


Note that this function may also return error codes from previous, asynchronous launches.


See also:

cuStreamDestroy, cuStreamCreateWithPriority, cuGreenCtxStreamCreate, cuStreamGetPriority,
cuStreamGetFlags, cuStreamGetDevice cuStreamWaitEvent, cuStreamQuery, cuStreamSynchronize,
cuStreamAddCallback, cudaStreamCreate, cudaStreamCreateWithFlags
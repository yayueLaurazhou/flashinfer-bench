# CUresult cuGreenCtxStreamCreate (CUstream *phStream, CUgreenCtx greenCtx, unsigned int flags, int priority)

Create a stream for use in the green context.

###### Parameters

**phStream**

  - Returned newly created stream
**greenCtx**

  - Green context for which to create the stream for


CUDA Driver API TRM-06703-001 _vRelease Version  |  586


Modules


**flags**

  - Flags for stream creation. CU_STREAM_NON_BLOCKING must be specified.
**priority**

  - Stream priority. Lower numbers represent higher priorities. See cuCtxGetStreamPriorityRange for
more information about meaningful stream priorities that can be passed.

###### Returns

CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE,
CUDA_ERROR_OUT_OF_MEMORY

###### Description

Creates a stream for use in the specified green context greenCtx and returns a handle in phStream.
The stream can be destroyed by calling cuStreamDestroy(). Note that the API ignores the context that
is current to the calling thread and creates a stream in the specified green context greenCtx.

The supported values for flags are:

CU_STREAM_NON_BLOCKING: This must be specified. It indicates that work running in the

###### **‣**

created stream may run concurrently with work in the default stream, and that the created stream
should perform no implicit synchronization with the default stream.

Specifying priority affects the scheduling priority of work in the stream. Priorities provide a hint
to preferentially run work with higher priority when possible, but do not preempt already-running work
or provide any other functional guarantee on execution order. priority follows a convention where
lower numbers represent higher priorities. '0' represents default priority. The range of meaningful
numerical priorities can be queried using cuCtxGetStreamPriorityRange. If the specified priority
is outside the numerical range returned by cuCtxGetStreamPriorityRange, it will automatically be
clamped to the lowest or the highest number in the range.


Note:

**‣** Note that this function may also return error codes from previous, asynchronous launches.

**‣** In the current implementation, only compute kernels launched in priority streams are affected by
the stream's priority. Stream priorities have no effect on host-to-device and device-to-host memory
operations.


See also:

cuStreamDestroy, cuGreenCtxCreate cuStreamCreate, cuStreamGetPriority,
cuCtxGetStreamPriorityRange, cuStreamGetFlags, cuStreamGetDevice, cuStreamWaitEvent,
cuStreamQuery, cuStreamSynchronize, cuStreamAddCallback, cudaStreamCreateWithPriority


CUDA Driver API TRM-06703-001 _vRelease Version  |  587


Modules
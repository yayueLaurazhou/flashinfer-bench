# CUresult cuStreamCreateWithPriority (CUstream *phStream, unsigned int flags, int priority)

Create a stream with the given priority.

###### Parameters

**phStream**

  - Returned newly created stream
**flags**

  - Flags for stream creation. See cuStreamCreate for a list of valid flags
**priority**

  - Stream priority. Lower numbers represent higher priorities. See cuCtxGetStreamPriorityRange for
more information about meaningful stream priorities that can be passed.

###### Returns

CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE,
CUDA_ERROR_OUT_OF_MEMORY


CUDA Driver API TRM-06703-001 _vRelease Version  |  338


Modules

###### Description

Creates a stream with the specified priority and returns a handle in phStream. This affects the
scheduling priority of work in the stream. Priorities provide a hint to preferentially run work with
higher priority when possible, but do not preempt already-running work or provide any other functional
guarantee on execution order.

priority follows a convention where lower numbers represent higher priorities. '0'
represents default priority. The range of meaningful numerical priorities can be queried using
cuCtxGetStreamPriorityRange. If the specified priority is outside the numerical range returned by
cuCtxGetStreamPriorityRange, it will automatically be clamped to the lowest or the highest number in
the range.


Note:



**‣** Note that this function may also return error codes from previous, asynchronous launches.



**‣** Stream priorities are supported only on GPUs with compute capability 3.5 or higher.



**‣** In the current implementation, only compute kernels launched in priority streams are affected by
the stream's priority. Stream priorities have no effect on host-to-device and device-to-host memory
operations.



See also:

cuStreamDestroy, cuStreamCreate, cuGreenCtxStreamCreate, cuStreamGetPriority,
cuCtxGetStreamPriorityRange, cuStreamGetFlags, cuStreamGetDevice, cuStreamWaitEvent,
cuStreamQuery, cuStreamSynchronize, cuStreamAddCallback, cudaStreamCreateWithPriority
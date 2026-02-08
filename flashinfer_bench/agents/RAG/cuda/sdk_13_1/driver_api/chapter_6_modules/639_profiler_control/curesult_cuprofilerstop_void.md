# CUresult cuProfilerStop (void)

Disable profiling.

###### Returns

CUDA_SUCCESS, CUDA_ERROR_INVALID_CONTEXT

###### Description

Disables profile collection by the active profiling tool for the current context. If profiling is already
disabled, then cuProfilerStop() has no effect.

cuProfilerStart and cuProfilerStop APIs are used to programmatically control the profiling granularity
by allowing profiling to be done only on selective pieces of code.


Note:


Note that this function may also return error codes from previous, asynchronous launches.


See also:

cuProfilerInitialize, cuProfilerStart, cudaProfilerStop
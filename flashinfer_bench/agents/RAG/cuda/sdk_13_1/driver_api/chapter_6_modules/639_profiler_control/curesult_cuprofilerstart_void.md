# CUresult cuProfilerStart (void)

Enable profiling.

###### Returns

CUDA_SUCCESS, CUDA_ERROR_INVALID_CONTEXT

###### Description

Enables profile collection by the active profiling tool for the current context. If profiling is already
enabled, then cuProfilerStart() has no effect.

cuProfilerStart and cuProfilerStop APIs are used to programmatically control the profiling granularity
by allowing profiling to be done only on selective pieces of code.


Note:


Note that this function may also return error codes from previous, asynchronous launches.


CUDA Driver API TRM-06703-001 _vRelease Version  |  598


Modules


See also:

cuProfilerInitialize, cuProfilerStop, cudaProfilerStart
# __host__cudaError_t cudaProfilerStart (void)

Enable profiling.

##### Returns

cudaSuccess


CUDA Runtime API vRelease Version  |  519


Modules

##### Description

Enables profile collection by the active profiling tool for the current context. If profiling is already
enabled, then cudaProfilerStart() has no effect.

cudaProfilerStart and cudaProfilerStop APIs are used to programmatically control the profiling
granularity by allowing profiling to be done only on selective pieces of code.


Note:


Note that this function may also return error codes from previous, asynchronous launches.


See also:

cudaProfilerStop, cuProfilerStart
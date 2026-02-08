# __host____device__cudaError_t cudaDeviceGetLimit (size_t *pValue, cudaLimit limit)

Return resource limits.

##### Parameters

**pValue**

  - Returned size of the limit
**limit**

  - Limit to query

##### Returns

cudaSuccess, cudaErrorUnsupportedLimit, cudaErrorInvalidValue

##### Description

Returns in *pValue the current size of limit. The following cudaLimit values are supported.

cudaLimitStackSize is the stack size in bytes of each GPU thread.

##### **‣**

cudaLimitPrintfFifoSize is the size in bytes of the shared FIFO used by the printf() device system

##### **‣**

call.
cudaLimitMallocHeapSize is the size in bytes of the heap used by the malloc() and free() device

##### **‣**

system calls.
cudaLimitDevRuntimeSyncDepth is the maximum grid depth at which a thread can isssue the

##### **‣**

device runtime call cudaDeviceSynchronize() to wait on child grid launches to complete. This
functionality is removed for devices of compute capability >= 9.0, and hence will return error
cudaErrorUnsupportedLimit on such devices.


CUDA Runtime API vRelease Version  |  16


Modules



cudaLimitDevRuntimePendingLaunchCount is the maximum number of outstanding device

##### **‣**

runtime launches.
cudaLimitMaxL2FetchGranularity is the L2 cache fetch granularity.

##### **‣**

cudaLimitPersistingL2CacheSize is the persisting L2 cache size in bytes.

##### **‣**

See also:

cudaDeviceSetLimit, cuCtxGetLimit
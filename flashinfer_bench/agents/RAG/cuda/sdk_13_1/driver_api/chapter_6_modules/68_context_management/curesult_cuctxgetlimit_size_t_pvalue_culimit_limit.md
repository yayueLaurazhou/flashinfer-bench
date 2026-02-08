# CUresult cuCtxGetLimit (size_t *pvalue, CUlimit limit)

Returns resource limits.

###### Parameters

**pvalue**

  - Returned size of limit


CUDA Driver API TRM-06703-001 _vRelease Version  |  129


Modules


**limit**

  - Limit to query

###### Returns

CUDA_SUCCESS, CUDA_ERROR_INVALID_VALUE, CUDA_ERROR_UNSUPPORTED_LIMIT

###### Description

Returns in *pvalue the current size of limit. The supported CUlimit values are:

CU_LIMIT_STACK_SIZE: stack size in bytes of each GPU thread.

###### **‣**

CU_LIMIT_PRINTF_FIFO_SIZE: size in bytes of the FIFO used by the printf() device system

###### **‣**

call.
CU_LIMIT_MALLOC_HEAP_SIZE: size in bytes of the heap used by the malloc() and free()

###### **‣**

device system calls.
CU_LIMIT_DEV_RUNTIME_SYNC_DEPTH: maximum grid depth at which a thread can issue

###### **‣**

the device runtime call cudaDeviceSynchronize() to wait on child grid launches to complete.
CU_LIMIT_DEV_RUNTIME_PENDING_LAUNCH_COUNT: maximum number of outstanding

###### **‣**

device runtime launches that can be made from this context.
CU_LIMIT_MAX_L2_FETCH_GRANULARITY: L2 cache fetch granularity.

###### **‣**

CU_LIMIT_PERSISTING_L2_CACHE_SIZE: Persisting L2 cache size in bytes

###### **‣**

Note:


Note that this function may also return error codes from previous, asynchronous launches.


See also:

cuCtxCreate, cuCtxDestroy, cuCtxGetApiVersion, cuCtxGetCacheConfig, cuCtxGetDevice,
cuCtxGetFlags, cuCtxPopCurrent, cuCtxPushCurrent, cuCtxSetCacheConfig, cuCtxSetLimit,
cuCtxSynchronize, cudaDeviceGetLimit
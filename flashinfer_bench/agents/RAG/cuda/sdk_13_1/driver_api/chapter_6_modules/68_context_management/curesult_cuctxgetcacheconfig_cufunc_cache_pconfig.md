# CUresult cuCtxGetCacheConfig (CUfunc_cache *pconfig)

Returns the preferred cache configuration for the current context.

###### Parameters

**pconfig**

  - Returned cache configuration

###### Returns

CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE

###### Description

On devices where the L1 cache and shared memory use the same hardware resources, this function
returns through pconfig the preferred cache configuration for the current context. This is only
a preference. The driver will use the requested configuration if possible, but it is free to choose a
different configuration if required to execute functions.

This will return a pconfig of CU_FUNC_CACHE_PREFER_NONE on devices where the size of
the L1 cache and shared memory are fixed.

The supported cache configurations are:

CU_FUNC_CACHE_PREFER_NONE: no preference for shared memory or L1 (default)

###### **‣**

CU_FUNC_CACHE_PREFER_SHARED: prefer larger shared memory and smaller L1 cache

###### **‣**

CU_FUNC_CACHE_PREFER_L1: prefer larger L1 cache and smaller shared memory

###### **‣**

CU_FUNC_CACHE_PREFER_EQUAL: prefer equal sized L1 cache and shared memory

###### **‣**

Note:


Note that this function may also return error codes from previous, asynchronous launches.


See also:

cuCtxCreate, cuCtxDestroy, cuCtxGetApiVersion, cuCtxGetDevice, cuCtxGetFlags, cuCtxGetLimit,
cuCtxPopCurrent, cuCtxPushCurrent, cuCtxSetCacheConfig, cuCtxSetLimit, cuCtxSynchronize,
cuFuncSetCacheConfig, cudaDeviceGetCacheConfig


CUDA Driver API TRM-06703-001 _vRelease Version  |  125


Modules
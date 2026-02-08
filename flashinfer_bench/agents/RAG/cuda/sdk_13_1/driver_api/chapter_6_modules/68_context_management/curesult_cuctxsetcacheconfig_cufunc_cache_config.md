# CUresult cuCtxSetCacheConfig (CUfunc_cache config)

Sets the preferred cache configuration for the current context.

###### Parameters

**config**

  - Requested cache configuration

###### Returns

CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE

###### Description

On devices where the L1 cache and shared memory use the same hardware resources, this sets through
config the preferred cache configuration for the current context. This is only a preference. The
driver will use the requested configuration if possible, but it is free to choose a different configuration
if required to execute the function. Any function preference set via cuFuncSetCacheConfig() or
cuKernelSetCacheConfig() will be preferred over this context-wide setting. Setting the context-wide
cache configuration to CU_FUNC_CACHE_PREFER_NONE will cause subsequent kernel launches
to prefer to not change the cache configuration unless required to launch the kernel.

This setting does nothing on devices where the size of the L1 cache and shared memory are fixed.

Launching a kernel with a different preference than the most recent preference setting may insert a
device-side synchronization point.

The supported cache configurations are:

CU_FUNC_CACHE_PREFER_NONE: no preference for shared memory or L1 (default)

###### **‣**

CU_FUNC_CACHE_PREFER_SHARED: prefer larger shared memory and smaller L1 cache

###### **‣**

CU_FUNC_CACHE_PREFER_L1: prefer larger L1 cache and smaller shared memory

###### **‣**

CUDA Driver API TRM-06703-001 _vRelease Version  |  134


Modules


CU_FUNC_CACHE_PREFER_EQUAL: prefer equal sized L1 cache and shared memory

###### **‣**

Note:


Note that this function may also return error codes from previous, asynchronous launches.


See also:

cuCtxCreate, cuCtxDestroy, cuCtxGetApiVersion, cuCtxGetCacheConfig, cuCtxGetDevice,
cuCtxGetFlags, cuCtxGetLimit, cuCtxPopCurrent, cuCtxPushCurrent, cuCtxSetLimit,
cuCtxSynchronize, cuFuncSetCacheConfig, cudaDeviceSetCacheConfig, cuKernelSetCacheConfig
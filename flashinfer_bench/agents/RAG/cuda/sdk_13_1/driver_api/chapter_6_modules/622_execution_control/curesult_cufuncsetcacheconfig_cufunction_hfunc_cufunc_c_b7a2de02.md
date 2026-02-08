# CUresult cuFuncSetCacheConfig (CUfunction hfunc, CUfunc_cache config)

Sets the preferred cache configuration for a device function.

###### Parameters

**hfunc**

  - Kernel to configure cache for
**config**

  - Requested cache configuration


CUDA Driver API TRM-06703-001 _vRelease Version  |  388


Modules

###### Returns

CUDA_SUCCESS, CUDA_ERROR_INVALID_VALUE, CUDA_ERROR_DEINITIALIZED,
CUDA_ERROR_NOT_INITIALIZED, CUDA_ERROR_INVALID_CONTEXT

###### Description

On devices where the L1 cache and shared memory use the same hardware resources, this sets
through config the preferred cache configuration for the device function hfunc. This is only
a preference. The driver will use the requested configuration if possible, but it is free to choose
a different configuration if required to execute hfunc. Any context-wide preference set via
cuCtxSetCacheConfig() will be overridden by this per-function setting unless the per-function setting is
CU_FUNC_CACHE_PREFER_NONE. In that case, the current context-wide setting will be used.

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

CU_FUNC_CACHE_PREFER_EQUAL: prefer equal sized L1 cache and shared memory

###### **‣**

Note:


Note that this function may also return error codes from previous, asynchronous launches.


See also:

cuCtxGetCacheConfig, cuCtxSetCacheConfig, cuFuncGetAttribute, cuLaunchKernel,
cudaFuncSetCacheConfig, cuKernelSetCacheConfig


CUDA Driver API TRM-06703-001 _vRelease Version  |  389


Modules
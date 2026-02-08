# __host____device__cudaError_t cudaDeviceGetCacheConfig (cudaFuncCache *pCacheConfig)

Returns the preferred cache configuration for the current device.

##### Parameters

**pCacheConfig**

  - Returned cache configuration


CUDA Runtime API vRelease Version  |  13


Modules

##### Returns

cudaSuccess

##### Description

On devices where the L1 cache and shared memory use the same hardware resources, this returns
through pCacheConfig the preferred cache configuration for the current device. This is only a
preference. The runtime will use the requested configuration if possible, but it is free to choose a
different configuration if required to execute functions.

This will return a pCacheConfig of cudaFuncCachePreferNone on devices where the size of the L1
cache and shared memory are fixed.

The supported cache configurations are:

cudaFuncCachePreferNone: no preference for shared memory or L1 (default)

##### **‣**

cudaFuncCachePreferShared: prefer larger shared memory and smaller L1 cache

##### **‣**

cudaFuncCachePreferL1: prefer larger L1 cache and smaller shared memory

##### **‣**

cudaFuncCachePreferEqual: prefer equal size L1 cache and shared memory

##### **‣**

See also:

cudaDeviceSetCacheConfig, cudaFuncSetCacheConfig ( C API), cudaFuncSetCacheConfig ( C++
API), cuCtxGetCacheConfig
# __host__cudaError_t cudaDeviceSetCacheConfig (cudaFuncCache cacheConfig)

Sets the preferred cache configuration for the current device.

##### Parameters

**cacheConfig**

  - Requested cache configuration

##### Returns

cudaSuccess

##### Description

On devices where the L1 cache and shared memory use the same hardware resources, this sets through
cacheConfig the preferred cache configuration for the current device. This is only a preference. The
runtime will use the requested configuration if possible, but it is free to choose a different configuration
if required to execute the function. Any function preference set via cudaFuncSetCacheConfig ( C API)
or cudaFuncSetCacheConfig ( C++ API) will be preferred over this device-wide setting. Setting the
device-wide cache configuration to cudaFuncCachePreferNone will cause subsequent kernel launches
to prefer to not change the cache configuration unless required to launch the kernel.

This setting does nothing on devices where the size of the L1 cache and shared memory are fixed.

Launching a kernel with a different preference than the most recent preference setting may insert a
device-side synchronization point.

The supported cache configurations are:

cudaFuncCachePreferNone: no preference for shared memory or L1 (default)

##### **‣**

cudaFuncCachePreferShared: prefer larger shared memory and smaller L1 cache

##### **‣**

cudaFuncCachePreferL1: prefer larger L1 cache and smaller shared memory

##### **‣**

cudaFuncCachePreferEqual: prefer equal size L1 cache and shared memory

##### **‣**

Note:

**‣** Note that this function may also return error codes from previous, asynchronous launches.


CUDA Runtime API vRelease Version  |  26


Modules





See also:

cudaDeviceGetCacheConfig, cudaFuncSetCacheConfig ( C API), cudaFuncSetCacheConfig ( C++
API), cuCtxSetCacheConfig
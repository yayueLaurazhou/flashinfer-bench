# template < class T > __host__cudaError_t cudaFuncSetCacheConfig (T *func, cudaFuncCache cacheConfig)

[C++ API] Sets the preferred cache configuration for a device function

##### Parameters

**func**

  - device function pointer
**cacheConfig**

  - Requested cache configuration

##### Returns

cudaSuccess, cudaErrorInvalidDeviceFunction

##### Description

On devices where the L1 cache and shared memory use the same hardware resources, this sets through
cacheConfig the preferred cache configuration for the function specified via func. This is only
a preference. The runtime will use the requested configuration if possible, but it is free to choose a
different configuration if required to execute func.

func must be a pointer to a function that executes on the device. The parameter specified by
func must be declared as a __global__ function. If the specified function does not exist, then
cudaErrorInvalidDeviceFunction is returned.

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

CUDA Runtime API vRelease Version  |  468


Modules


cudaLaunchKernel ( C++ API), cudaFuncSetCacheConfig ( C API), cudaFuncGetAttributes
( C++ API), cudaSetDoubleForDevice, cudaSetDoubleForHost, cudaThreadGetCacheConfig,
cudaThreadSetCacheConfig
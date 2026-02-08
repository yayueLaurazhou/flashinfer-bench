# __host__cudaError_t cudaFuncSetCacheConfig (const void *func, cudaFuncCache cacheConfig)

Sets the preferred cache configuration for a device function.

##### Parameters

**func**

  - Device function symbol
**cacheConfig**

  - Requested cache configuration

##### Returns

cudaSuccess, cudaErrorInvalidDeviceFunction

##### Description

On devices where the L1 cache and shared memory use the same hardware resources, this sets through
cacheConfig the preferred cache configuration for the function specified via func. This is only
a preference. The runtime will use the requested configuration if possible, but it is free to choose a
different configuration if required to execute func.

func is a device function symbol and must be declared as a __global__ function. If the specified
function does not exist, then cudaErrorInvalidDeviceFunction is returned. For templated functions, pass
the function symbol as follows: func_name<template_arg_0,...,template_arg_N>

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

**‣** Use of a string naming a function as the func parameter was deprecated in CUDA 4.1 and
removed in CUDA 5.0.


CUDA Runtime API vRelease Version  |  100


Modules





See also:

cudaFuncSetCacheConfig ( C++ API), cudaFuncGetAttributes ( C API), cudaLaunchKernel ( C API),
cuFuncSetCacheConfig
# __host__cudaError_t cudaFuncSetSharedMemConfig (const void *func, cudaSharedMemConfig config)

Sets the shared memory configuration for a device function.

##### Parameters

**func**

  - Device function symbol
**config**

  - Requested shared memory configuration

##### Returns

cudaSuccess, cudaErrorInvalidDeviceFunction, cudaErrorInvalidValue,

##### Description

Deprecated

On devices with configurable shared memory banks, this function will force all subsequent launches of
the specified device function to have the given shared memory bank size configuration. On any given
launch of the function, the shared memory configuration of the device will be temporarily changed if
needed to suit the function's preferred configuration. Changes in shared memory configuration between
subsequent launches of functions, may introduce a device side synchronization point.

Any per-function setting of shared memory bank size set via cudaFuncSetSharedMemConfig will
override the device wide setting set by cudaDeviceSetSharedMemConfig.

Changing the shared memory bank size will not increase shared memory usage or affect occupancy of
kernels, but may have major effects on performance. Larger bank sizes will allow for greater potential
bandwidth to shared memory, but will change what kinds of accesses to shared memory will result in
bank conflicts.

This function will do nothing on devices with fixed shared memory bank size.

For templated functions, pass the function symbol as follows:
func_name<template_arg_0,...,template_arg_N>

The supported bank configurations are:


CUDA Runtime API vRelease Version  |  110


Modules


cudaSharedMemBankSizeDefault: use the device's shared memory configuration when launching

##### **‣**

this function.
cudaSharedMemBankSizeFourByte: set shared memory bank width to be four bytes natively when

##### **‣**

launching this function.
cudaSharedMemBankSizeEightByte: set shared memory bank width to be eight bytes natively

##### **‣**

when launching this function.





See also:

cudaDeviceSetSharedMemConfig, cudaDeviceGetSharedMemConfig, cudaDeviceSetCacheConfig,
cudaDeviceGetCacheConfig, cudaFuncSetCacheConfig, cuFuncSetSharedMemConfig
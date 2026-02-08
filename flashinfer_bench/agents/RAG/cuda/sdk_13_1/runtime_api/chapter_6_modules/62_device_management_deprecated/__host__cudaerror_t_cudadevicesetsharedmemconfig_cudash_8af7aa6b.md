# __host__cudaError_t cudaDeviceSetSharedMemConfig (cudaSharedMemConfig config)

Sets the shared memory configuration for the current device.

##### Parameters

**config**

  - Requested cache configuration

##### Returns

cudaSuccess, cudaErrorInvalidValue

##### Description

Deprecated

On devices with configurable shared memory banks, this function will set the shared memory bank size
which is used for all subsequent kernel launches. Any per-function setting of shared memory set via
cudaFuncSetSharedMemConfig will override the device wide setting.

Changing the shared memory configuration between launches may introduce a device side
synchronization point.

Changing the shared memory bank size will not increase shared memory usage or affect occupancy of
kernels, but may have major effects on performance. Larger bank sizes will allow for greater potential
bandwidth to shared memory, but will change what kinds of accesses to shared memory will result in
bank conflicts.

This function will do nothing on devices with fixed shared memory bank size.

The supported bank configurations are:

cudaSharedMemBankSizeDefault: set bank width the device default (currently, four bytes)

##### **‣**

cudaSharedMemBankSizeFourByte: set shared memory bank width to be four bytes natively.

##### **‣**

cudaSharedMemBankSizeEightByte: set shared memory bank width to be eight bytes natively.

##### **‣**

See also:


CUDA Runtime API vRelease Version  |  45


Modules


cudaDeviceSetCacheConfig, cudaDeviceGetCacheConfig, cudaDeviceGetSharedMemConfig,
cudaFuncSetCacheConfig, cuCtxSetSharedMemConfig
# __host____device__cudaError_t cudaDeviceGetSharedMemConfig (cudaSharedMemConfig *pConfig)

Returns the shared memory configuration for the current device.

##### Parameters

**pConfig**

  - Returned cache configuration

##### Returns

cudaSuccess, cudaErrorInvalidValue

##### Description

Deprecated

This function will return in pConfig the current size of shared memory banks on the current device.
On devices with configurable shared memory banks, cudaDeviceSetSharedMemConfig can be used to
change this setting, so that all subsequent kernel launches will by default use the new bank size. When
cudaDeviceGetSharedMemConfig is called on devices without configurable shared memory, it will
return the fixed bank size of the hardware.

The returned bank configurations can be either:

cudaSharedMemBankSizeFourByte - shared memory bank width is four bytes.

##### **‣**

cudaSharedMemBankSizeEightByte - shared memory bank width is eight bytes.

##### **‣**

See also:

cudaDeviceSetCacheConfig, cudaDeviceGetCacheConfig, cudaDeviceSetSharedMemConfig,
cudaFuncSetCacheConfig, cuCtxGetSharedMemConfig


CUDA Runtime API vRelease Version  |  44


Modules
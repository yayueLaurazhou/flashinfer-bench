# CUresult cuFuncSetSharedMemConfig (CUfunction hfunc, CUsharedconfig config)

Sets the shared memory configuration for a device function.

###### Parameters

**hfunc**

  - kernel to be given a shared memory config
**config**

  - requested shared memory configuration

###### Returns

CUDA_SUCCESS, CUDA_ERROR_INVALID_VALUE, CUDA_ERROR_DEINITIALIZED,
CUDA_ERROR_NOT_INITIALIZED, CUDA_ERROR_INVALID_CONTEXT

###### Description

Deprecated

On devices with configurable shared memory banks, this function will force all subsequent launches of
the specified device function to have the given shared memory bank size configuration. On any given
launch of the function, the shared memory configuration of the device will be temporarily changed if
needed to suit the function's preferred configuration. Changes in shared memory configuration between
subsequent launches of functions, may introduce a device side synchronization point.

Any per-function setting of shared memory bank size set via cuFuncSetSharedMemConfig will
override the context wide setting set with cuCtxSetSharedMemConfig.

Changing the shared memory bank size will not increase shared memory usage or affect occupancy of
kernels, but may have major effects on performance. Larger bank sizes will allow for greater potential
bandwidth to shared memory, but will change what kinds of accesses to shared memory will result in
bank conflicts.

This function will do nothing on devices with fixed shared memory bank size.

The supported bank configurations are:

CU_SHARED_MEM_CONFIG_DEFAULT_BANK_SIZE: use the context's shared memory

###### **‣**

configuration when launching this function.
CU_SHARED_MEM_CONFIG_FOUR_BYTE_BANK_SIZE: set shared memory bank width to

###### **‣**

be natively four bytes when launching this function.
CU_SHARED_MEM_CONFIG_EIGHT_BYTE_BANK_SIZE: set shared memory bank width to

###### **‣**

be natively eight bytes when launching this function.


Note:


CUDA Driver API TRM-06703-001 _vRelease Version  |  405


Modules


Note that this function may also return error codes from previous, asynchronous launches.


See also:

cuCtxGetCacheConfig, cuCtxSetCacheConfig, cuCtxGetSharedMemConfig,
cuCtxSetSharedMemConfig, cuFuncGetAttribute, cuLaunchKernel, cudaFuncSetSharedMemConfig
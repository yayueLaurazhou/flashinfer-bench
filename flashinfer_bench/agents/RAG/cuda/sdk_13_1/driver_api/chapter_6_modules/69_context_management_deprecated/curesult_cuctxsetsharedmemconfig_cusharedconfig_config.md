# CUresult cuCtxSetSharedMemConfig (CUsharedconfig config)

Sets the shared memory configuration for the current context.

###### Parameters

**config**

  - requested shared memory configuration

###### Returns

CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE

###### Description

Deprecated

On devices with configurable shared memory banks, this function will set the context's shared memory
bank size which is used for subsequent kernel launches.

Changed the shared memory configuration between launches may insert a device side synchronization
point between those launches.

Changing the shared memory bank size will not increase shared memory usage or affect occupancy of
kernels, but may have major effects on performance. Larger bank sizes will allow for greater potential


CUDA Driver API TRM-06703-001 _vRelease Version  |  143


Modules


bandwidth to shared memory, but will change what kinds of accesses to shared memory will result in
bank conflicts.

This function will do nothing on devices with fixed shared memory bank size.

The supported bank configurations are:

CU_SHARED_MEM_CONFIG_DEFAULT_BANK_SIZE: set bank width to the default initial

###### **‣**

setting (currently, four bytes).
CU_SHARED_MEM_CONFIG_FOUR_BYTE_BANK_SIZE: set shared memory bank width to

###### **‣**

be natively four bytes.
CU_SHARED_MEM_CONFIG_EIGHT_BYTE_BANK_SIZE: set shared memory bank width to

###### **‣**

be natively eight bytes.


Note:


Note that this function may also return error codes from previous, asynchronous launches.


See also:

cuCtxCreate, cuCtxDestroy, cuCtxGetApiVersion, cuCtxGetCacheConfig,
cuCtxGetDevice, cuCtxGetFlags, cuCtxGetLimit, cuCtxPopCurrent, cuCtxPushCurrent,
cuCtxSetLimit, cuCtxSynchronize, cuCtxGetSharedMemConfig, cuFuncSetCacheConfig,
cudaDeviceSetSharedMemConfig
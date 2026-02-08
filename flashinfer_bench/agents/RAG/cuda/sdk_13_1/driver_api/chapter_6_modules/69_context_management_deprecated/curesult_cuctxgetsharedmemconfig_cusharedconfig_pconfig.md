# CUresult cuCtxGetSharedMemConfig (CUsharedconfig *pConfig)

Returns the current shared memory configuration for the current context.

###### Parameters

**pConfig**

  - returned shared memory configuration

###### Returns

CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE

###### Description

Deprecated

This function will return in pConfig the current size of shared memory banks in the current context.
On devices with configurable shared memory banks, cuCtxSetSharedMemConfig can be used to
change this setting, so that all subsequent kernel launches will by default use the new bank size. When


CUDA Driver API TRM-06703-001 _vRelease Version  |  142


Modules


cuCtxGetSharedMemConfig is called on devices without configurable shared memory, it will return
the fixed bank size of the hardware.

The returned bank configurations can be either:

CU_SHARED_MEM_CONFIG_FOUR_BYTE_BANK_SIZE: shared memory bank width is four

###### **‣**

bytes.
CU_SHARED_MEM_CONFIG_EIGHT_BYTE_BANK_SIZE: shared memory bank width will

###### **‣**

eight bytes.


Note:


Note that this function may also return error codes from previous, asynchronous launches.


See also:

cuCtxCreate, cuCtxDestroy, cuCtxGetApiVersion, cuCtxGetCacheConfig,
cuCtxGetDevice, cuCtxGetFlags, cuCtxGetLimit, cuCtxPopCurrent, cuCtxPushCurrent,
cuCtxSetLimit, cuCtxSynchronize, cuCtxGetSharedMemConfig, cuFuncSetCacheConfig,
cudaDeviceGetSharedMemConfig
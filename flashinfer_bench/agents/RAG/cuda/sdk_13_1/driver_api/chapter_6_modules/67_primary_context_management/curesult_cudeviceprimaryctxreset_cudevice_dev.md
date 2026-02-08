# CUresult cuDevicePrimaryCtxReset (CUdevice dev)

Destroy all allocations and reset all state on the primary context.

###### Parameters

**dev**

  - Device for which primary context is destroyed

###### Returns

CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
CUDA_ERROR_INVALID_DEVICE, CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE

###### Description

Explicitly destroys and cleans up all resources associated with the current device in the current process.

Note that it is responsibility of the calling function to ensure that no other module in the process is
using the device any more. For that reason it is recommended to use cuDevicePrimaryCtxRelease()
in most cases. However it is safe for other modules to call cuDevicePrimaryCtxRelease() even after
resetting the device. Resetting the primary context does not release it, an application that has retained
the primary context should explicitly release its usage.


Note:


Note that this function may also return error codes from previous, asynchronous launches.


See also:

cuDevicePrimaryCtxRetain, cuDevicePrimaryCtxRelease, cuCtxGetApiVersion,
cuCtxGetCacheConfig, cuCtxGetDevice, cuCtxGetFlags, cuCtxGetLimit, cuCtxPopCurrent,
cuCtxPushCurrent, cuCtxSetCacheConfig, cuCtxSetLimit, cuCtxSynchronize, cudaDeviceReset


CUDA Driver API TRM-06703-001 _vRelease Version  |  116


Modules
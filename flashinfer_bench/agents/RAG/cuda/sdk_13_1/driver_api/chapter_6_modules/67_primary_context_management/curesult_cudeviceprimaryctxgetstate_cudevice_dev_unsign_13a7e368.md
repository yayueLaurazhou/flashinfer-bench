# CUresult cuDevicePrimaryCtxGetState (CUdevice dev, unsigned int *flags, int *active)

Get the state of the primary context.

###### Parameters

**dev**

  - Device to get primary context flags for
**flags**

  - Pointer to store flags
**active**

  - Pointer to store context state; 0 = inactive, 1 = active


CUDA Driver API TRM-06703-001 _vRelease Version  |  114


Modules

###### Returns

CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
CUDA_ERROR_INVALID_DEVICE, CUDA_ERROR_INVALID_VALUE,

###### Description

Returns in *flags the flags for the primary context of dev, and in *active whether it is active. See
cuDevicePrimaryCtxSetFlags for flag values.


Note:


Note that this function may also return error codes from previous, asynchronous launches.


See also:

cuDevicePrimaryCtxSetFlags, cuCtxGetFlags, cuCtxSetFlags, cudaGetDeviceFlags
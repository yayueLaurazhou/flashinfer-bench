# CUresult cuDeviceUnregisterAsyncNotification (CUdevice device, CUasyncCallbackHandle callback)

Unregisters an async notification callback.

###### Parameters

**device**

  - The device from which to remove callback.
**callback**

  - The callback instance to unregister from receiving async notifications.

###### Returns

CUDA_SUCCESS, CUDA_ERROR_NOT_SUPPORTED, CUDA_ERROR_INVALID_DEVICE,
CUDA_ERROR_INVALID_VALUE, CUDA_ERROR_NOT_PERMITTED,
CUDA_ERROR_UNKNOWN

###### Description

Unregisters callback so that the corresponding callback function will stop receiving async
notifications.


Note:


Note that this function may also return error codes from previous, asynchronous launches.


See also:

cuDeviceRegisterAsyncNotification
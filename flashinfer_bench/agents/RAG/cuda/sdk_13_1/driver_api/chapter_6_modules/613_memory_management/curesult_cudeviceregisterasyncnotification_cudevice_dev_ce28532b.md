# CUresult cuDeviceRegisterAsyncNotification (CUdevice device, CUasyncCallback callbackFunc, void *userData, CUasyncCallbackHandle *callback)

Registers a callback function to receive async notifications.

###### Parameters

**device**

  - The device on which to register the callback
**callbackFunc**

  - The function to register as a callback
**userData**

  - A generic pointer to user data. This is passed into the callback function.
**callback**

  - A handle representing the registered callback instance

###### Returns

CUDA_SUCCESS, CUDA_ERROR_NOT_SUPPORTED, CUDA_ERROR_INVALID_DEVICE,
CUDA_ERROR_INVALID_VALUE, CUDA_ERROR_NOT_PERMITTED,
CUDA_ERROR_UNKNOWN

###### Description

Registers callbackFunc to receive async notifications.

The userData parameter is passed to the callback function at async notification time. Likewise,
callback is also passed to the callback function to distinguish between multiple registered callbacks.

The callback function being registered should be designed to return quickly (~10ms). Any long running
tasks should be queued for execution on an application thread.

Callbacks may not call cuDeviceRegisterAsyncNotification or cuDeviceUnregisterAsyncNotification.
Doing so will result in CUDA_ERROR_NOT_PERMITTED. Async notification callbacks execute in
an undefined order and may be serialized.

Returns in *callback a handle representing the registered callback instance.


Note:


Note that this function may also return error codes from previous, asynchronous launches.


See also:

cuDeviceUnregisterAsyncNotification


CUDA Driver API TRM-06703-001 _vRelease Version  |  189


Modules
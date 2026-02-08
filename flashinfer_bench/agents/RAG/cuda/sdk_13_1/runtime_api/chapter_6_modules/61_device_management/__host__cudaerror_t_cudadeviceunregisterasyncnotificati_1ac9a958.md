# __host__cudaError_t cudaDeviceUnregisterAsyncNotification (int device, cudaAsyncCallbackHandle_t callback)

Unregisters an async notification callback.

##### Parameters

**device**

  - The device from which to remove callback.
**callback**

  - The callback instance to unregister from receiving async notifications.

##### Returns

cudaSuccess cudaErrorNotSupported cudaErrorInvalidDevice cudaErrorInvalidValue
cudaErrorNotPermitted cudaErrorUnknown

##### Description

Unregisters callback so that the corresponding callback function will stop receiving async
notifications.


Note:


Note that this function may also return error codes from previous, asynchronous launches.


CUDA Runtime API vRelease Version  |  30


Modules



See also:

cudaDeviceRegisterAsyncNotification
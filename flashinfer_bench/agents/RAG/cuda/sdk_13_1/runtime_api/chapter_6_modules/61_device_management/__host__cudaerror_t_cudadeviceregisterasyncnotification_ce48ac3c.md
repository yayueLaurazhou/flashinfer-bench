# __host__cudaError_t cudaDeviceRegisterAsyncNotification (int device, cudaAsyncCallback callbackFunc, void *userData, cudaAsyncCallbackHandle_t *callback)

Registers a callback function to receive async notifications.

##### Parameters

**device**

  - The device on which to register the callback
**callbackFunc**

  - The function to register as a callback
**userData**

  - A generic pointer to user data. This is passed into the callback function.
**callback**

  - A handle representing the registered callback instance

##### Returns

cudaSuccess cudaErrorNotSupported cudaErrorInvalidDevice cudaErrorInvalidValue
cudaErrorNotPermitted cudaErrorUnknown

##### Description

Registers callbackFunc to receive async notifications.

The userData parameter is passed to the callback function at async notification time. Likewise,
callback is also passed to the callback function to distinguish between multiple registered callbacks.

The callback function being registered should be designed to return quickly (~10ms). Any long running
tasks should be queued for execution on an application thread.

Callbacks may not call cudaDeviceRegisterAsyncNotification or
cudaDeviceUnregisterAsyncNotification. Doing so will result in cudaErrorNotPermitted. Async
notification callbacks execute in an undefined order and may be serialized.

Returns in *callback a handle representing the registered callback instance.


CUDA Runtime API vRelease Version  |  24


Modules


Note:


Note that this function may also return error codes from previous, asynchronous launches.


See also:

cudaDeviceUnregisterAsyncNotification
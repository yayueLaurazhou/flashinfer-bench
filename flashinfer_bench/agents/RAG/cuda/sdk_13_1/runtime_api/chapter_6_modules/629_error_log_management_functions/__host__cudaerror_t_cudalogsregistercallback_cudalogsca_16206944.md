# __host__cudaError_t cudaLogsRegisterCallback (cudaLogsCallback_t callbackFunc, void *userData, cudaLogsCallbackHandle *callback_out)

Register a callback function to receive error log messages.

##### Parameters

**callbackFunc**

  - The function to register as a callback
**userData**

  - A generic pointer to user data. This is passed into the callback function.
**callback_out**

  - Optional location to store the callback handle after it is registered

##### Returns

cudaSuccess, cudaErrorInvalidValue,


CUDA Runtime API vRelease Version  |  319


Modules
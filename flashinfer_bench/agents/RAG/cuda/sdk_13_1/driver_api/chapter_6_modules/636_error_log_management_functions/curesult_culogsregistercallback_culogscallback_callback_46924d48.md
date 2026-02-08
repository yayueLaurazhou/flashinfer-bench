# CUresult cuLogsRegisterCallback (CUlogsCallback callbackFunc, void *userData, CUlogsCallbackHandle *callback_out)

Register a callback function to receive error log messages.

###### Parameters

**callbackFunc**

  - The function to register as a callback
**userData**

  - A generic pointer to user data. This is passed into the callback function.
**callback_out**

  - Optional location to store the callback handle after it is registered

###### Returns

CUDA_SUCCESS CUDA_ERROR_INVALID_VALUE
# CUresult cuStreamDestroy (CUstream hStream)

Destroys a stream.

###### Parameters

**hStream**

  - Stream to destroy

###### Returns

CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE,
CUDA_ERROR_INVALID_HANDLE

###### Description

Destroys the stream specified by hStream.


CUDA Driver API TRM-06703-001 _vRelease Version  |  339


Modules


In case the device is still doing work in the stream hStream when cuStreamDestroy() is called,
the function will return immediately and the resources associated with hStream will be released
automatically once the device has completed all work in hStream.


Note:


Note that this function may also return error codes from previous, asynchronous launches.


See also:

cuStreamCreate, cuStreamWaitEvent, cuStreamQuery, cuStreamSynchronize, cuStreamAddCallback,
cudaStreamDestroy
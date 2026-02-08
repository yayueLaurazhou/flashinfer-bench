# CUresult cuEventDestroy (CUevent hEvent)

Destroys an event.

###### Parameters

**hEvent**

  - Event to destroy

###### Returns

CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_HANDLE

###### Description

Destroys the event specified by hEvent.

An event may be destroyed before it is complete (i.e., while cuEventQuery() would return
CUDA_ERROR_NOT_READY). In this case, the call does not block on completion of the event, and
any associated resources will automatically be released asynchronously at completion.


Note:


Note that this function may also return error codes from previous, asynchronous launches.


See also:

cuEventCreate, cuEventRecord, cuEventQuery, cuEventSynchronize, cuEventElapsedTime,
cudaEventDestroy
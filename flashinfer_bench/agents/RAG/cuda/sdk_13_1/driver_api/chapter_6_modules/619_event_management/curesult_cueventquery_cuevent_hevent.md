# CUresult cuEventQuery (CUevent hEvent)

Queries an event's status.

###### Parameters

**hEvent**

  - Event to query


CUDA Driver API TRM-06703-001 _vRelease Version  |  357


Modules

###### Returns

CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
CUDA_ERROR_INVALID_HANDLE, CUDA_ERROR_INVALID_VALUE,
CUDA_ERROR_NOT_READY

###### Description

Queries the status of all work currently captured by hEvent. See cuEventRecord() for details on what
is captured by an event.

Returns CUDA_SUCCESS if all captured work has been completed, or
CUDA_ERROR_NOT_READY if any captured work is incomplete.

For the purposes of Unified Memory, a return value of CUDA_SUCCESS is equivalent to having
called cuEventSynchronize().


Note:


Note that this function may also return error codes from previous, asynchronous launches.


See also:

cuEventCreate, cuEventRecord, cuEventSynchronize, cuEventDestroy, cuEventElapsedTime,
cudaEventQuery
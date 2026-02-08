# CUresult cuStreamQuery (CUstream hStream)

Determine status of a compute stream.

###### Parameters

**hStream**

  - Stream to query status of

###### Returns

CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_HANDLE,
CUDA_ERROR_NOT_READY


CUDA Driver API TRM-06703-001 _vRelease Version  |  349


Modules

###### Description

Returns CUDA_SUCCESS if all operations in the stream specified by hStream have completed, or
CUDA_ERROR_NOT_READY if not.

For the purposes of Unified Memory, a return value of CUDA_SUCCESS is equivalent to having
called cuStreamSynchronize().





See also:

cuStreamCreate, cuStreamWaitEvent, cuStreamDestroy, cuStreamSynchronize, cuStreamAddCallback,
cudaStreamQuery
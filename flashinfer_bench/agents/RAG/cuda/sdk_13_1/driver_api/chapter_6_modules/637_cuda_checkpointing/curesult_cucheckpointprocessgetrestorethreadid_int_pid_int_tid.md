# CUresult cuCheckpointProcessGetRestoreThreadId (int pid, int *tid)

Returns the restore thread ID for a CUDA process.

###### Parameters

**pid**

  - The process ID of the CUDA process
**tid**

  - Returned restore thread ID

###### Returns

CUDA_SUCCESS CUDA_ERROR_INVALID_VALUE CUDA_ERROR_NOT_INITIALIZED
CUDA_ERROR_NOT_SUPPORTED

###### Description

Returns in *tid the thread ID of the CUDA restore thread for the process specified by pid.


CUDA Driver API TRM-06703-001 _vRelease Version  |  594


Modules
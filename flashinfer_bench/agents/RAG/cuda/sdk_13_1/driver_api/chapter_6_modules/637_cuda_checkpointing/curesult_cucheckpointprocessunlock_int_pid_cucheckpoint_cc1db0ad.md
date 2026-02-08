# CUresult cuCheckpointProcessUnlock (int pid, CUcheckpointUnlockArgs *args)

Unlock a CUDA process to allow CUDA API calls.

###### Parameters

**pid**

  - The process ID of the CUDA process


CUDA Driver API TRM-06703-001 _vRelease Version  |  596


Modules


**args**

  - Optional unlock operation arguments

###### Returns

CUDA_SUCCESS CUDA_ERROR_INVALID_VALUE CUDA_ERROR_NOT_INITIALIZED
CUDA_ERROR_ILLEGAL_STATE CUDA_ERROR_NOT_SUPPORTED

###### Description

Unlocks a process specified by pid allowing it to resume making CUDA API calls. Process must be in
the LOCKED state.

Upon successful return the process will be in the RUNNING state.
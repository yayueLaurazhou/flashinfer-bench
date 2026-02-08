# CUresult cuCheckpointProcessLock (int pid, CUcheckpointLockArgs *args)

Lock a running CUDA process.

###### Parameters

**pid**

  - The process ID of the CUDA process
**args**

  - Optional lock operation arguments

###### Returns

CUDA_SUCCESS CUDA_ERROR_INVALID_VALUE CUDA_ERROR_NOT_INITIALIZED
CUDA_ERROR_ILLEGAL_STATE CUDA_ERROR_NOT_SUPPORTED
CUDA_ERROR_NOT_READY

###### Description

Lock the CUDA process specified by pid which will block further CUDA API calls. Process must be
in the RUNNING state in order to lock.

Upon successful return the process will be in the LOCKED state.


CUDA Driver API TRM-06703-001 _vRelease Version  |  595


Modules


If timeoutMs is specified and the timeout is reached the process will be left in the RUNNING state
upon return.
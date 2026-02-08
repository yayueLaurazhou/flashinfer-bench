# CUresult cuCheckpointProcessGetState (int pid, CUprocessState *state)

Returns the process state of a CUDA process.

###### Parameters

**pid**

  - The process ID of the CUDA process
**state**

  - Returned CUDA process state

###### Returns

CUDA_SUCCESS CUDA_ERROR_INVALID_VALUE CUDA_ERROR_NOT_INITIALIZED
CUDA_ERROR_NOT_SUPPORTED

###### Description

Returns in *state the current state of the CUDA process specified by pid.
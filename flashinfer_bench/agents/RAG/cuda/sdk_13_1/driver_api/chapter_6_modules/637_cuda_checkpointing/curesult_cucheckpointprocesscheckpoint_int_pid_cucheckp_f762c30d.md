# CUresult cuCheckpointProcessCheckpoint (int pid, CUcheckpointCheckpointArgs *args)

Checkpoint a CUDA process's GPU memory contents.

###### Parameters

**pid**

  - The process ID of the CUDA process
**args**

  - Optional checkpoint operation arguments

###### Returns

CUDA_SUCCESS CUDA_ERROR_INVALID_VALUE CUDA_ERROR_NOT_INITIALIZED
CUDA_ERROR_ILLEGAL_STATE CUDA_ERROR_NOT_SUPPORTED

###### Description

Checkpoints a CUDA process specified by pid that is in the LOCKED state. The GPU memory
contents will be brought into host memory and all underlying references will be released. Process must
be in the LOCKED state to checkpoint.

Upon successful return the process will be in the CHECKPOINTED state.
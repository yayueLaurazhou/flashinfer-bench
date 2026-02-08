# CUresult cuCheckpointProcessRestore (int pid, CUcheckpointRestoreArgs *args)

Restore a CUDA process's GPU memory contents from its last checkpoint.

###### Parameters

**pid**

  - The process ID of the CUDA process
**args**

  - Optional restore operation arguments

###### Returns

CUDA_SUCCESS CUDA_ERROR_INVALID_VALUE CUDA_ERROR_NOT_INITIALIZED
CUDA_ERROR_ILLEGAL_STATE CUDA_ERROR_NOT_SUPPORTED

###### Description

Restores a CUDA process specified by pid from its last checkpoint. Process must be in the
CHECKPOINTED state to restore.

GPU UUID pairs can be specified in args to remap the process old GPUs onto new GPUs. The GPU
to restore onto needs to have enough memory and be of the same chip type as the old GPU. If an array
of GPU UUID pairs is specified, it must contain every checkpointed GPU.

Upon successful return the process will be in the LOCKED state.

CUDA process restore requires persistence mode to be enabled or cuInit to have been called before
execution.


See also:

cuInit
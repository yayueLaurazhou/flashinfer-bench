# enum CUprocessState

CUDA Process States

###### Values

**CU_PROCESS_STATE_RUNNING = 0**
Default process state
**CU_PROCESS_STATE_LOCKED**
CUDA API locks are taken so further CUDA API calls will block
**CU_PROCESS_STATE_CHECKPOINTED**
Application memory contents have been checkpointed and underlying allocations and device
handles have been released
**CU_PROCESS_STATE_FAILED**
Application entered an uncorrectable error during the checkpoint/restore process
# enum CUcomputemode

Compute Modes

###### Values

**CU_COMPUTEMODE_DEFAULT = 0**
Default compute mode (Multiple contexts allowed per device)
**CU_COMPUTEMODE_PROHIBITED = 2**
Compute-prohibited mode (No contexts can be created on this device at this time)
**CU_COMPUTEMODE_EXCLUSIVE_PROCESS = 3**
Compute-exclusive-process mode (Only one context used by a single process can be present on this
device at a time)
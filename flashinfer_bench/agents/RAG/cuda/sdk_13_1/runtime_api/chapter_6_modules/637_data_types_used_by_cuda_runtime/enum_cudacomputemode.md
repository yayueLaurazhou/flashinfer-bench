# enum cudaComputeMode

CUDA device compute modes

##### Values

**cudaComputeModeDefault = 0**
Default compute mode (Multiple threads can use cudaSetDevice() with this device)
**cudaComputeModeExclusive = 1**
Compute-exclusive-thread mode (Only one thread in one process will be able to use
cudaSetDevice() with this device)
**cudaComputeModeProhibited = 2**
Compute-prohibited mode (No threads can use cudaSetDevice() with this device)
**cudaComputeModeExclusiveProcess = 3**
Compute-exclusive-process mode (Many threads in one process will be able to use cudaSetDevice()
with this device)
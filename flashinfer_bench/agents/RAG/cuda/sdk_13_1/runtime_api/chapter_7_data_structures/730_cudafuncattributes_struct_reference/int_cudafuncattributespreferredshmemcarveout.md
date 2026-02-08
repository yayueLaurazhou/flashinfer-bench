# int cudaFuncAttributes::preferredShmemCarveout

On devices where the L1 cache and shared memory use the same hardware resources, this sets
the shared memory carveout preference, in percent of the maximum shared memory. Refer to
cudaDevAttrMaxSharedMemoryPerMultiprocessor. This is only a hint, and the driver can choose a
different ratio if required to execute the function. See cudaFuncSetAttribute
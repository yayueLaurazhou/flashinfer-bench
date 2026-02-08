# 13.6.1.1.3. Synchronization (CDP1)

##### 13.6.1.1.3. Synchronization (CDP1)[](#synchronization-cdp1 "Permalink to this headline")

See [Synchronization](#dynamic-parallelism-synchronization), above, for CDP2 version of document.

Warning

Explicit synchronization with child kernels from a parent block (i.e. using `cudaDeviceSynchronize()` in device code) is deprecated in CUDA 11.6, removed for compute\_90+ compilation, and is slated for full removal in a future CUDA release.

CUDA runtime operations from any thread, including kernel launches, are visible across a thread block. This means that an invoking thread in the parent grid may perform synchronization on the grids launched by that thread, by other threads in the thread block, or on streams created within the same thread block. Execution of a thread block is not considered complete until all launches by all threads in the block have completed. If all threads in a block exit before all child launches have completed, a synchronization operation will automatically be triggered.
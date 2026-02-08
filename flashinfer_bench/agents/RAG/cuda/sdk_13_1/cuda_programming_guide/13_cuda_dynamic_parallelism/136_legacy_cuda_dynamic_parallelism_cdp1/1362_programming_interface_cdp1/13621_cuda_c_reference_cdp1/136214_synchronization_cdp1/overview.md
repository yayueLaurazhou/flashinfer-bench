# 13.6.2.1.4. Synchronization (CDP1)

##### 13.6.2.1.4. Synchronization (CDP1)[](#synchronization-programming-interface-cdp1 "Permalink to this headline")

See [Synchronization](#synchronization-programming-interface), above, for CDP2 version of document.

Warning

Explicit synchronization with child kernels from a parent block (i.e. using `cudaDeviceSynchronize()` in device code) is deprecated in CUDA 11.6, removed for compute\_90+ compilation, and is slated for full removal in a future CUDA release.

The `cudaDeviceSynchronize()` function will synchronize on all work launched by any thread in the thread-block up to the point where `cudaDeviceSynchronize()` was called. Note that `cudaDeviceSynchronize()` may be called from within divergent code (see [Block Wide Synchronization (CDP1)](#block-wide-synchronization-cdp1)).

It is up to the program to perform sufficient additional inter-thread synchronization, for example via a call to `__syncthreads()`, if the calling thread is intended to synchronize with child grids invoked from other threads.
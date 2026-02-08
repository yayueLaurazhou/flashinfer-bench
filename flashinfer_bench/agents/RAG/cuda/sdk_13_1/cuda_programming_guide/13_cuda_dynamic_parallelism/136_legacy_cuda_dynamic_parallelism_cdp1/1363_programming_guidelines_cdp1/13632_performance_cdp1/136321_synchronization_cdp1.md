# 13.6.3.2.1. Synchronization (CDP1)

##### 13.6.3.2.1. Synchronization (CDP1)[](#synchronization-performance-cdp1 "Permalink to this headline")

See [CUDA Dynamic Parallelism](#cuda-dynamic-parallelism), above, for CDP2 version of document.

Warning

Explicit synchronization with child kernels from a parent block (such as using `cudaDeviceSynchronize()` in device code) is deprecated in CUDA 11.6, removed for compute\_90+ compilation, and is slated for full removal in a future CUDA release.

Synchronization by one thread may impact the performance of other threads in the same *Thread Block*, even when those other threads do not call `cudaDeviceSynchronize()` themselves. This impact will depend upon the underlying implementation. In general the implicit synchronization of child kernels done when a thread block ends is more efficient compared to calling `cudaDeviceSynchronize()` explicitly. It is therefore recommended to only call `cudaDeviceSynchronize()` if it is needed to synchronize with a child kernel before a thread block ends.
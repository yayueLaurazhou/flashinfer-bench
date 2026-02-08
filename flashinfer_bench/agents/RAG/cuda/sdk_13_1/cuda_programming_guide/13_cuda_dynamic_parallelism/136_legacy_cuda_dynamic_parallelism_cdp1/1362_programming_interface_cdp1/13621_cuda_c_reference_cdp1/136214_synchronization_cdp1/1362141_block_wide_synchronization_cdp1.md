# 13.6.2.1.4.1. Block Wide Synchronization (CDP1)

###### 13.6.2.1.4.1. Block Wide Synchronization (CDP1)[](#block-wide-synchronization-cdp1 "Permalink to this headline")

See [CUDA Dynamic Parallelism](#cuda-dynamic-parallelism), above, for CDP2 version of document.

The `cudaDeviceSynchronize()` function does not imply intra-block synchronization. In particular, without explicit synchronization via a `__syncthreads()` directive the calling thread can make no assumptions about what work has been launched by any thread other than itself. For example if multiple threads within a block are each launching work and synchronization is desired for all this work at once (perhaps because of event-based dependencies), it is up to the program to guarantee that this work is submitted by all threads before calling `cudaDeviceSynchronize()`.

Because the implementation is permitted to synchronize on launches from any thread in the block, it is quite possible that simultaneous calls to `cudaDeviceSynchronize()` by multiple threads will drain all work in the first call and then have no effect for the later calls.
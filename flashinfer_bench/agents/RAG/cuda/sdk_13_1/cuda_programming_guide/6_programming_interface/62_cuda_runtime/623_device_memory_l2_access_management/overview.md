# 6.2.3. Device Memory L2 Access Management

### 6.2.3. Device Memory L2 Access Management[](#device-memory-l2-access-management "Permalink to this headline")

When a CUDA kernel accesses a data region in the global memory repeatedly, such data accesses can be considered to be *persisting*. On the other hand, if the data is only accessed once, such data accesses can be considered to be *streaming*.

Starting with CUDA 11.0, devices of compute capability 8.0 and above have the capability to influence persistence of data in the L2 cache, potentially providing higher bandwidth and lower latency accesses to global memory.
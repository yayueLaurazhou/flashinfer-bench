# 13.6.1.1.2. Scope of CUDA Primitives (CDP1)

##### 13.6.1.1.2. Scope of CUDA Primitives (CDP1)[](#scope-of-cuda-primitives-cdp1 "Permalink to this headline")

See [Scope of CUDA Primitives](#scope-of-cuda-primitives-cdp2), above, for CDP2 version of document.

On both host and device, the CUDA runtime offers an API for launching kernels, for waiting for launched work to complete, and for tracking dependencies between launches via streams and events. On the host system, the state of launches and the CUDA primitives referencing streams and events are shared by all threads within a process; however processes execute independently and may not share CUDA objects.

A similar hierarchy exists on the device: launched kernels and CUDA objects are visible to all threads in a thread block, but are independent between thread blocks. This means for example that a stream may be created by one thread and used by any other thread in the same thread block, but may not be shared with threads in any other thread block.
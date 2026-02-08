# 13.2.1.2. Scope of CUDA Primitives

#### 13.2.1.2. Scope of CUDA Primitives[](#scope-of-cuda-primitives "Permalink to this headline")

On both host and device, the CUDA runtime offers an API for launching kernels and for tracking dependencies between launches via streams and events. On the host system, the state of launches and the CUDA primitives referencing streams and events are shared by all threads within a process; however processes execute independently and may not share CUDA objects.

On the device, launched kernels and CUDA objects are visible to all threads in a grid. This means, for example, that a stream may be created by one thread and used by any other thread in the grid.
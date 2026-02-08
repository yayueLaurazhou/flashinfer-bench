# 5.3. Memory Hierarchy

## 5.3. Memory Hierarchy[](#memory-hierarchy "Permalink to this headline")

CUDA threads may access data from multiple memory spaces during their execution as illustrated by [Figure 6](#memory-hierarchy-memory-hierarchy-figure). Each thread has private local memory. Each thread block has shared memory visible to all threads of the block and with the same lifetime as the block. Thread blocks in a thread block cluster can perform read, write, and atomics operations on each other’s shared memory. All threads have access to the same global memory.

There are also two additional read-only memory spaces accessible by all threads: the constant and texture memory spaces. The global, constant, and texture memory spaces are optimized for different memory usages (see [Device Memory Accesses](#device-memory-accesses)). Texture memory also offers different addressing modes, as well as data filtering, for some specific data formats (see [Texture and Surface Memory](#texture-and-surface-memory)).

The global, constant, and texture memory spaces are persistent across kernel launches by the same application.

![Memory Hierarchy](_images/memory-hierarchy.png)


Figure 6 Memory Hierarchy[](#memory-hierarchy-memory-hierarchy-figure "Permalink to this image")
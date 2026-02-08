# 16.4. Optimized Memory Reuse

## 16.4. Optimized Memory Reuse[](#optimized-memory-reuse "Permalink to this headline")

CUDA reuses memory in two ways:

* Virtual and physical memory reuse within a graph is based on virtual address assignment, like in the stream ordered allocator.
* Physical memory reuse between graphs is done with virtual aliasing: different graphs can map the same physical memory to their unique virtual addresses.
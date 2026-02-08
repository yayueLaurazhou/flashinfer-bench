# 24.1.2.4. Difference between Unified Memory and Mapped Memory

#### 24.1.2.4. Difference between Unified Memory and Mapped Memory[](#difference-between-unified-memory-and-mapped-memory "Permalink to this headline")

The main difference between Unified Memory and [Mapped Memory](#mapped-memory) is that
CUDA Mapped Memory does not guarantee that all kinds of memory accesses (for example atomics) are supported on all systems,
while Unified Memory does. The limited set of memory operations that are guaranteed to be portably supported by CUDA Mapped Memory
is available on more systems than Unified Memory.
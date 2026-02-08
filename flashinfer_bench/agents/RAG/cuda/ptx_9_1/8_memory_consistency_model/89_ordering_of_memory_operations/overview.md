# 8.9. Ordering of memory operations

## 8.9. [Ordering of memory operations](https://docs.nvidia.com/cuda/parallel-thread-execution/#ordering-memory-operations)[](https://docs.nvidia.com/cuda/parallel-thread-execution/#ordering-memory-operations "Permalink to this headline")

The sequence of operations performed by each thread is captured as *program order* while *memory
synchronization* across threads is captured as *causality order*. The visibility of the side-effects
of memory operations to other memory operations is captured as *communication order*. The memory
consistency model defines contradictions that are disallowed between communication order on the one
hand, and *causality order* and *program order* on the other.
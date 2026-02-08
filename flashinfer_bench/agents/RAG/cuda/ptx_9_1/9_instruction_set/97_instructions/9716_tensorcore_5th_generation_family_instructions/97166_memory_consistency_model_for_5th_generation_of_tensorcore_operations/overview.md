# 9.7.16.6. Memory Consistency Model for 5th generation of TensorCore operations

#### 9.7.16.6. [Memory Consistency Model for 5th generation of TensorCore operations](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-memory-consistency-model)[](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-memory-consistency-model "Permalink to this headline")

Ordering of `tcgen05` instructions is described in terms of two key concepts:

1. Pipelined tcgen05 instructions
2. Specialized tcgen05-specific inter-thread synchronization mechanisms.

These concepts combine to form four canonical synchronization patterns, as described further below.
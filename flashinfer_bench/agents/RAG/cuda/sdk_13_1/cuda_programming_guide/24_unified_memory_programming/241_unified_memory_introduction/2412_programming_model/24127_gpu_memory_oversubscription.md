# 24.1.2.7. GPU Memory Oversubscription

#### 24.1.2.7. GPU Memory Oversubscription[](#gpu-memory-oversubscription "Permalink to this headline")

Unified Memory enables applications to *oversubscribe* the memory of any individual processor:
in other words they can allocate and share arrays larger than
the memory capacity of any individual processor in the system,
enabling among others out-of-core processing of datasets that do not fit within
a single GPU, without adding significant complexity to the programming model.
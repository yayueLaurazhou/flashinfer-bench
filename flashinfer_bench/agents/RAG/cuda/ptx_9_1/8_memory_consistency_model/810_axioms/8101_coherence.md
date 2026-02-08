# 8.10.1. Coherence

### 8.10.1. [Coherence](https://docs.nvidia.com/cuda/parallel-thread-execution/#coherence-axiom)[](https://docs.nvidia.com/cuda/parallel-thread-execution/#coherence-axiom "Permalink to this headline")

If a write W precedes an *overlapping* write W’ in *causality order*, then W must precede W’ in
*coherence order*.
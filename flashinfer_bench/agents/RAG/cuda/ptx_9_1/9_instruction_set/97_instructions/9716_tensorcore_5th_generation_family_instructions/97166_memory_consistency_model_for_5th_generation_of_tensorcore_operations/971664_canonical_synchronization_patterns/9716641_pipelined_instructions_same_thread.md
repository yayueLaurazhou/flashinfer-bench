# 9.7.16.6.4.1. Pipelined instructions, same thread

###### 9.7.16.6.4.1. [Pipelined instructions, same thread](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-memory-consistency-model-canonical-sync-patterns-pipelined-same-thread)[](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-memory-consistency-model-canonical-sync-patterns-pipelined-same-thread "Permalink to this headline")

In this pattern, no explicit ordering mechanism is needed and the ordering guarantee is
provided by the pipelined instruction pairing.

Example:

```
tcgen05.mma
tcgen05.mma (same shape and accumulator)
```

Copy to clipboard

The two instructions will be executed in program order.
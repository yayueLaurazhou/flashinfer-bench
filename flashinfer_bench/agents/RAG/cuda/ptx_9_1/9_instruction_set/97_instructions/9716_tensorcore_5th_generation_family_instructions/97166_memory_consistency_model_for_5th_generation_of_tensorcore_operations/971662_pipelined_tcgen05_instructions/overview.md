# 9.7.16.6.2. Pipelined tcgen05 Instructions

##### 9.7.16.6.2. [Pipelined tcgen05 Instructions](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-memory-consistency-model-pipelined-instructions)[](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-memory-consistency-model-pipelined-instructions "Permalink to this headline")

The asynchronous `tcgen05` operations may execute and complete in a different order than they
were issued. However, some specific pairs of the asynchronous `tcgen05` instructions form
`tcgen05` pipelines, where in the two asynchronous operations are guaranteed to execute in
the same order as the instructions that issued them. The specific pairings are as follows:

1. `tcgen05.mma.cta_group::N` -> `tcgen05.mma.cta_group::N` (same N and accumulator and shape)
2. `tcgen05.copy.cta_group::N` -> `tcgen05.mma.cta_group::N` (same N)
3. `tcgen05.shift.cta_group::N` -> `tcgen05.mma.cta_group::N` (same N)
4. `tcgen05.shift.cta_group::N` -> `tcgen05.cp.4x256b.cta_group::N` (same N)
5. `tcgen05.mma.cta_group::N` -> `tcgen05.shift.cta_group::N` (same N)
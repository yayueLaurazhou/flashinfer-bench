# tcgen05-memory-consistency-model-mbarrier-completion

###### 9.7.16.6.2.1.1. [mbarrier based completion mechanism](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-memory-consistency-model-mbarrier-completion)[](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-memory-consistency-model-mbarrier-completion "Permalink to this headline")

Completion of the following instruction’s asynchronous operations is observed
through the mbarrier based waiting mechanism:

1. `tcgen05.mma`
2. `tcgen05.cp`
3. `tcgen05.shift`

`tcgen05.commit` is used to track the completion of the above asynchronous instructions.

Following are the implicitly pipelined `tcgen05` instruction pairing that uses mbarrier
based completion mechanism:

* `tcgen05.mma.cta_group::N` -> `tcgen05.commit.cta_group::N` (same N)
* `tcgen05.cp.cta_group::N` -> `tcgen05.commit.cta_group::N` (same N)
* `tcgen05.shift.cta_group::N` -> `tcgen05.commit.cta_group::N` (same N)
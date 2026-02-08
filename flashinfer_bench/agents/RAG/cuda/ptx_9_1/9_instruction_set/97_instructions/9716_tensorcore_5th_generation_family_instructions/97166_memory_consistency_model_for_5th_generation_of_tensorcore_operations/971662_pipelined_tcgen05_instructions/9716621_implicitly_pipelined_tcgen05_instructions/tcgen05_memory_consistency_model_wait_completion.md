# tcgen05-memory-consistency-model-wait-completion

###### 9.7.16.6.2.1.2. [`tcgen05.wait` instruction based completion mechanism](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-memory-consistency-model-wait-completion)[](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-memory-consistency-model-wait-completion "Permalink to this headline")

Completion of the following instruction’s asynchronous operations is observed through
`tcgen05.wait` based waiting mechanism:

1. `tcgen05.ld`
2. `tcgen05.st`

`tcgen05.wait::ld` and `tcgen05.wait::st` is used to track the completion of the
`tcgen05.ld` and `tcgen05.st` asynchronous instructions.

Following are the implicitly pipelined `tcgen05` instruction pairing that uses
`tcgen05.wait` based completion mechanism:

* `tcgen05.ld` -> `tcgen05.wait::ld`
* `tcgen05.st` -> `tcgen05.wait::st`
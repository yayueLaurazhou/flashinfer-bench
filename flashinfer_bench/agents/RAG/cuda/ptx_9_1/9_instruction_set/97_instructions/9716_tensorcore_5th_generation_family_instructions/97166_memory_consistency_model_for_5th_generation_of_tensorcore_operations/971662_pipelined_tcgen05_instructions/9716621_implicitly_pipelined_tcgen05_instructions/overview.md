# 9.7.16.6.2.1. Implicitly pipelined tcgen05 Instructions

###### 9.7.16.6.2.1. [Implicitly pipelined tcgen05 Instructions](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-memory-consistency-model-pipelined-instructions-implicit)[](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-memory-consistency-model-pipelined-instructions-implicit "Permalink to this headline")

Instructions `tcgen05.commit` and `tcgen05.wait` are implicitly pipelined with respect
to previously issued `tcgen05.{mma,cp,shift}` and `tcgen05.{ld,st}` instructions
respectively that they track from the same thread.
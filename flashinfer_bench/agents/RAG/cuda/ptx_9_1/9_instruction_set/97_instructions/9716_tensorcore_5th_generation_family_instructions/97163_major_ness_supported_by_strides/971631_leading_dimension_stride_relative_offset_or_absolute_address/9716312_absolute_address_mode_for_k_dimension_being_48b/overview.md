# 9.7.16.3.1.2. Absolute address mode for K dimension being 48B

###### 9.7.16.3.1.2. [Absolute address mode for K dimension being 48B](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-leading-dimension-byte-offset-absolute-address)[](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-leading-dimension-byte-offset-absolute-address "Permalink to this headline")

The `tcgen05.mma` instruction with *K-dimension* of 48B would overflow the 128B
shared memory boundary if the data is packed contiguously.

In this case, the absolute address mode can be used to break up the data in the
shared memory into two chunks such that both these chunks are laid out within
the aligned 128-byte address boundary.
The leading dimension absolute address can point to the second data chunk in the shared memory.
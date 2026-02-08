# tcgen05-matrix-fragments-shape-3232b

###### 9.7.16.2.3.1.1. [Matrix fragments for shape .32x32b](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-matrix-fragments-shape-3232b)[](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-matrix-fragments-shape-3232b "Permalink to this headline")

A `tcgen05{.ld,.st}.32x32b` instruction has the following data vector register.

| Fragment | Elements (low to high) |
| --- | --- |
| A vector expression containing `.num` number of `.b32` registers as mentioned in the [Table 49](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-num-shapes-ld). | r0, r1, … |

A warp executing `tcgen05{.ld,.st}.32x32b` will access 32 lanes of the Tensor Memory.
It loads from or stores to each of the lane (32 \* .num)-bits of data as shown in
[Figure 183](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-mma-fragment-3232b).

![_images/tcgen05-mma-fragment-3232b.png](./ptx_files/tcgen05-mma-fragment-3232b.png)


Figure 183 Matrix Fragment for shape .32x32b[](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-mma-fragment-3232b "Permalink to this image")
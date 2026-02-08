# tcgen05-matrix-fragments-shape-16256b

###### 9.7.16.2.3.1.4. [Matrix fragments for shape .16x256b](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-matrix-fragments-shape-16256b)[](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-matrix-fragments-shape-16256b "Permalink to this headline")

A `tcgen05{.ld,.st}.16x256b` instruction has the following data vector register.

| Fragment | Elements (low to high) |
| --- | --- |
| A vector expression containing `.num` number of `.b32` registers as mentioned in the [Table 49](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-num-shapes-ld). | r0, r1, r2, r3, … |

A warp executing `tcgen05{.ld,.st}.16x256b` will access 16 lanes of the Tensor Memory.
It loads from or stores to each of the lane (256 \* .num)-bits of data as shown in
[Figure 186](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-mma-fragment-16256b).

![_images/tcgen05-mma-fragment-16256b.png](./ptx_files/tcgen05-mma-fragment-16256b.png)


Figure 186 Matrix Fragment for shape .16x256b[](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-mma-fragment-16256b "Permalink to this image")
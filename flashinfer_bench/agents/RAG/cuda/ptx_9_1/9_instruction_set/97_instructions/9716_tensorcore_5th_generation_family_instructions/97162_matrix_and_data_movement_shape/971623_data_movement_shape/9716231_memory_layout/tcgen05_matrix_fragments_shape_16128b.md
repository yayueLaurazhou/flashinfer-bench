# tcgen05-matrix-fragments-shape-16128b

###### 9.7.16.2.3.1.3. [Matrix fragments for shape .16x128b](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-matrix-fragments-shape-16128b)[](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-matrix-fragments-shape-16128b "Permalink to this headline")

A `tcgen05{.ld,.st}.16x128b` instruction has the following data vector register.

| Fragment | Elements (low to high) |
| --- | --- |
| A vector expression containing `.num` number of `.b32` registers as mentioned in the [Table 49](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-num-shapes-ld). | r0, r1, … |

A warp executing `tcgen05{.ld,.st}.16x128b` will access 16 lanes of the Tensor Memory.
It loads from or stores to each of the lane (128 \* .num)-bits of data as shown in
[Figure 185](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-mma-fragment-16128b).

![_images/tcgen05-mma-fragment-16128b.png](./ptx_files/tcgen05-mma-fragment-16128b.png)


Figure 185 Matrix Fragment for shape .16x128b[](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-mma-fragment-16128b "Permalink to this image")
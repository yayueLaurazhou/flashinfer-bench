# tcgen05-matrix-fragments-shape-1632b2

###### 9.7.16.2.3.1.5. [Matrix fragments for shape .16x32bx2](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-matrix-fragments-shape-1632b2)[](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-matrix-fragments-shape-1632b2 "Permalink to this headline")

A `tcgen05{.ld,.st}.16x32bx2` instruction has the following data vector register.

| Fragment | Elements (low to high) |
| --- | --- |
| A vector expression containing `.num` number of `.b32` registers as mentioned in the [Table 49](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-num-shapes-ld). | r0, r1, … |

A warp executing `tcgen05{.ld,.st}.16x32bx2` will access 16 lanes of the Tensor Memory.
It loads from or stores to each of the lane (32 \* .num)-bits of data as shown in
[Figure 187](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-mma-fragment-1632b2).

![_images/tcgen05-mma-fragment-1632b2.png](./ptx_files/tcgen05-mma-fragment-1632b2.png)


Figure 187 Matrix Fragment for shape .16x32bx2[](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-mma-fragment-1632b2 "Permalink to this image")
# 9.7.16.10.7. Block Scaling for tcgen05.mma.sync

##### 9.7.16.10.7. [Block Scaling for `tcgen05.mma.sync`](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-block-scaling)[](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-block-scaling "Permalink to this headline")

The `tcgen05.mma` instructions with the following `.kind` qualifier:

* `.kind::mxf8f6f4`
* `.kind::mxf4`
* `.kind::mxf4nvf4`

perform matrix multiplication with block scaling. This operation has the following form:

`(A * scale_A)  * (B * scale_B) + D`

where `scale_A` and `scale_B` are matrices residing in [Tensor Memory](https://docs.nvidia.com/cuda/parallel-thread-execution/#tensor-memory).

For a `scale_A` matrix of shape *M x SFA\_N*, each row of matrix `A` is divided into
*SFA\_N* number of chunks and each chunk of a row is multiplied with the corresponding
element in the *SF\_A* of the same row.

Similarly, for a `scale_B` matrix of shape *SFB\_M x N*, each column of matrix `B` is
divided into the *SFB\_M* number of chunks and each chunk of a column is multiplied with
the corresponding element in the *SF\_B* of the same column.

Scale factors for `A` and `B` matrices need to be duplicated to all 32 lane partitions
of tensor memory.

[Figure 230](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-mma-block-scaling) shows an example of `tcgen05.mma` with block scaling of
`scale_vec::2X`.

![_images/tcgen05-mma-block-scaling.png](./ptx_files/tcgen05-mma-block-scaling.png)


Figure 230 `tcgen05.mma` with block scaling of `scale_vec::2X`[](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-mma-block-scaling "Permalink to this image")
# tcgen05-mma-scale-factor-a-layout-1x

###### 9.7.16.10.7.2.1. [Layout of the Scale Factor A Matrix for scale\_vec::1X/block32 with K=32/K=64](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-mma-scale-factor-a-layout-1x)[](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-mma-scale-factor-a-layout-1x "Permalink to this headline")

There is one scale factor per row of the `A` matrix with block size as 32 and the scale factor must be provided in
1-byte aligned sub-column of the Tensor Memory. *SFA\_ID* specifies the byte offset in the
Tensor Memory word that must be used for the scale factor matrix.
[Figure 231](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-mma-scale-factor-a-1x-dig) shows which sub-columns get selected for
different values of *SFA\_ID*.

![_images/tcgen05-mma-scale-factor-a-1x-dig.png](./ptx_files/tcgen05-mma-scale-factor-a-1x-dig.png)


Figure 231 Layout of scale factor A matrix with scale\_vec::1X/block32 with K=32/K=64[](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-mma-scale-factor-a-1x-dig "Permalink to this image")

For example, if *SFA\_ID* is 0, then all the green columns are selected to form the scale factor
matrix. Similarly, *SFA\_ID* values of 1, 2 and 3 would select the blue, yellow, and red columns,
respectively.
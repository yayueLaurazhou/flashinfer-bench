# tcgen05-mma-scale-factor-b-layout-1x

###### 9.7.16.10.7.3.1. [Layout of the Scale Factor B Matrix for scale\_vec::1X/block32 with K=32/K=64](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-mma-scale-factor-b-layout-1x)[](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-mma-scale-factor-b-layout-1x "Permalink to this headline")

There is one scale factor per row of the `B` matrix with block size as 32 and the scale factor must be provided in
1-byte aligned sub-column of the Tensor Memory. *SFB\_ID* specifies the byte offset in the
Tensor Memory word that must be used for the scale factor matrix.
[Figure 240](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-mma-scale-factor-b-1x-dig) shows which sub-columns get selected for
different values of *SFB\_ID*.

![_images/tcgen05-mma-scale-factor-b-1x-dig.png](./ptx_files/tcgen05-mma-scale-factor-b-1x-dig.png)


Figure 240 Layout of scale factor B matrix with scale\_vec::1X/block32 with K=32/K=64[](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-mma-scale-factor-b-1x-dig "Permalink to this image")

For example, if *SFB\_ID* is 0, then all the green columns are selected to form the scale factor
matrix. Similarly, *SFB\_ID* values of 1, 2 and 3 would select the blue, yellow, and red columns, respectively.
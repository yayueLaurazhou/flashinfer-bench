# tcgen05-mma-scale-factor-a-layout-2x

###### 9.7.16.10.7.2.2. [Layout of the Scale Factor A Matrix for scale\_vec::2X/block32 with K=64/K=128](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-mma-scale-factor-a-layout-2x)[](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-mma-scale-factor-a-layout-2x "Permalink to this headline")

There are two scale factors per row of the `A` matrix with block size as 32 and the scale factor must be provided in
2-byte aligned sub-column of the Tensor Memory. *SFA\_ID* specifies the half word offset in the
Tensor Memory word that must be used for the scale factor matrix.
[Figure 232](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-mma-scale-factor-a-2x-dig) shows which sub-columns gets selected for different
values of *SFA\_ID*.

![_images/tcgen05-mma-scale-factor-a-2x-dig.png](./ptx_files/tcgen05-mma-scale-factor-a-2x-dig.png)


Figure 232 Layout of scale factor A matrix with scale\_vec::2X/block32 with K=64/K=128[](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-mma-scale-factor-a-2x-dig "Permalink to this image")

For example, if *SFA\_ID* is 0, then all the green columns are selected to form the scale factor
matrix. Similarly, if *SFA\_ID* is 2, then all of the blue columns are selected to form the scale
factor matrix.
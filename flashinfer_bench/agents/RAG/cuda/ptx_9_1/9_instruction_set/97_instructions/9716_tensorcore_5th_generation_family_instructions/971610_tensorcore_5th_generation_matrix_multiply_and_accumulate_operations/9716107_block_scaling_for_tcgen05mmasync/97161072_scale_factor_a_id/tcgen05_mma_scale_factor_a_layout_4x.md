# tcgen05-mma-scale-factor-a-layout-4x

###### 9.7.16.10.7.2.3. [Layout of the Scale Factor A Matrix for scale\_vec::4X/block16 with K=64/K=128](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-mma-scale-factor-a-layout-4x)[](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-mma-scale-factor-a-layout-4x "Permalink to this headline")

There are four scale factors per row of the `A` matrix with block size as 16 and the scale factor must be provided in
4-byte aligned sub-column of the Tensor Memory. The *SFA\_ID* value must be 0 and this specifies
that all of the columns (in green) will be used for the scale factor matrix.
[Figure 233](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-mma-scale-factor-a-4x-dig) shows which sub-columns gets selected for different
values of *SFA\_ID*.

![_images/tcgen05-mma-scale-factor-a-4x-dig.png](./ptx_files/tcgen05-mma-scale-factor-a-4x-dig.png)


Figure 233 Layout of scale factor A matrix with scale\_vec::4X/block16 with K=64/K=128[](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-mma-scale-factor-a-4x-dig "Permalink to this image")
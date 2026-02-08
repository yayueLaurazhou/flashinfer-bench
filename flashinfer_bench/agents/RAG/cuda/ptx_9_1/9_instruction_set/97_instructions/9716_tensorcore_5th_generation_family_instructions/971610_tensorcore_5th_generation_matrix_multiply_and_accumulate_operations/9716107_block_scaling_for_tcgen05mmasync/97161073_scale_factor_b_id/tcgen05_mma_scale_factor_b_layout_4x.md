# tcgen05-mma-scale-factor-b-layout-4x

###### 9.7.16.10.7.3.3. [Layout of the Scale Factor B Matrix for scale\_vec::4X/block16 with K=64/K=128](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-mma-scale-factor-b-layout-4x)[](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-mma-scale-factor-b-layout-4x "Permalink to this headline")

There are four scale factors per row of the `B` matrix with block size as 16 and the scale factor must be provided in
4-byte aligned sub-column of the Tensor Memory. The *SFB\_ID* value must be 0 and this specifies
that all of the columns (in green) will be used for the scale factor matrix.
[Figure 242](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-mma-scale-factor-b-4x-dig) shows which sub-columns get selected for
different values of *SFB\_ID*.

![_images/tcgen05-mma-scale-factor-b-4x-dig.png](./ptx_files/tcgen05-mma-scale-factor-b-4x-dig.png)


Figure 242 Layout of scale factor B matrix with scale\_vec::4X/block16 with K=64/K=128[](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-mma-scale-factor-b-4x-dig "Permalink to this image")
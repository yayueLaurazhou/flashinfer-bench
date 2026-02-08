# tcgen05-mma-scale-factor-b-layout-2x

###### 9.7.16.10.7.3.2. [Layout of the Scale Factor B Matrix for scale\_vec::2X/block32 with K=64/K=128](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-mma-scale-factor-b-layout-2x)[](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-mma-scale-factor-b-layout-2x "Permalink to this headline")

There are two scale factors per row of the `B` matrix with block size as 32 and the scale factor must be provided in
2-byte aligned sub-column of the Tensor Memory. *SFB\_ID* specifies the half word offset in the
Tensor Memory word that must be used for the scale factor matrix.
[Figure 241](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-mma-scale-factor-b-2x-dig) shows which sub-columns get selected for
different values of *SFB\_ID*.

![_images/tcgen05-mma-scale-factor-b-2x-dig.png](./ptx_files/tcgen05-mma-scale-factor-b-2x-dig.png)


Figure 241 Layout of scale factor B matrix with scale\_vec::2X/block32 with K=64/K=128[](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-mma-scale-factor-b-2x-dig "Permalink to this image")

For example, if *SFB\_ID* is 0, then all the green columns are selected to form the scale factor
matrix. Similarly, if *SFB\_ID* is 2, then all of the blue columns are selected to form the scale
factor matrix.
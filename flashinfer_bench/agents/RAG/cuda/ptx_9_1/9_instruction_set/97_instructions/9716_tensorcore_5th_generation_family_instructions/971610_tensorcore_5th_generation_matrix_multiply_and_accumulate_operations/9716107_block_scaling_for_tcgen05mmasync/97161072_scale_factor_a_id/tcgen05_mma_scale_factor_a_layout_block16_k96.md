# tcgen05-mma-scale-factor-a-layout-block16-k96

###### 9.7.16.10.7.2.5. [Layout of the Scale Factor A Matrix for block16 with K=96 (Semantically equivalent to scale\_vec::6X)](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-mma-scale-factor-a-layout-block16-k96)[](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-mma-scale-factor-a-layout-block16-k96 "Permalink to this headline")

There are six scale factors per row of the `A` matrix with block size as 16 and the scale
factor must be provided in 4-byte aligned sub-column of the Tensor Memory. *SFA\_ID* specifies
the byte offset in the Tensor Memory word that must be used for the scale factor matrix.
[Figure 238](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-mma-scale-factor-a-block16-k96-dig1) and [Figure 239](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-mma-scale-factor-a-block16-k96-dig2)
show which sub-columns get selected for different values of *SFA\_ID*.

![_images/tcgen05-mma-scale-factor-a-block16-k96-dig1.png](./ptx_files/tcgen05-mma-scale-factor-a-block16-k96-dig1.png)


Figure 238 Layout of scale factor A matrix with block16 with K=96 with SFA\_ID=00[](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-mma-scale-factor-a-block16-k96-dig1 "Permalink to this image")


![_images/tcgen05-mma-scale-factor-a-block16-k96-dig2.png](./ptx_files/tcgen05-mma-scale-factor-a-block16-k96-dig2.png)


Figure 239 Layout of scale factor A matrix with block16 with K=96 with SFA\_ID=10[](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-mma-scale-factor-a-block16-k96-dig2 "Permalink to this image")

For example, if *SFA\_ID* is 0, then all the green columns are selected to form the scale factor
matrix. Similarly, if *SFA\_ID* is 2, then all of the blue columns are selected to form the scale
factor matrix.
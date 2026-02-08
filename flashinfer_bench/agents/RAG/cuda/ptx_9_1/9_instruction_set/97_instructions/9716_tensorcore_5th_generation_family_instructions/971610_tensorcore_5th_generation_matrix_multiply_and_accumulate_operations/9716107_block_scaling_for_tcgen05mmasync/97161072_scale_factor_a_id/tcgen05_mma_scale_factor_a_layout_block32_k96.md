# tcgen05-mma-scale-factor-a-layout-block32-k96

###### 9.7.16.10.7.2.4. [Layout of the Scale Factor A Matrix for block32 with K=96 (Semantically equivalent to scale\_vec::3X)](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-mma-scale-factor-a-layout-block32-k96)[](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-mma-scale-factor-a-layout-block32-k96 "Permalink to this headline")

There are three scale factors per row of the `A` matrix with block size as 32 and the scale
factor must be provided in 4-byte aligned sub-column of the Tensor Memory. *SFA\_ID* specifies
the byte offset in the Tensor Memory word that must be used for the scale factor matrix.
[Figure 234](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-mma-scale-factor-a-block32-k96-dig1), [Figure 235](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-mma-scale-factor-a-block32-k96-dig2),
[Figure 236](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-mma-scale-factor-a-block32-k96-dig3) and [Figure 237](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-mma-scale-factor-a-block32-k96-dig4)
show which sub-columns get selected for different values of *SFA\_ID*.

![_images/tcgen05-mma-scale-factor-a-block32-k96-dig1.png](./ptx_files/tcgen05-mma-scale-factor-a-block32-k96-dig1.png)


Figure 234 Layout of scale factor A matrix with block32 with K=96 with SFA\_ID=00[](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-mma-scale-factor-a-block32-k96-dig1 "Permalink to this image")


![_images/tcgen05-mma-scale-factor-a-block32-k96-dig2.png](./ptx_files/tcgen05-mma-scale-factor-a-block32-k96-dig2.png)


Figure 235 Layout of scale factor A matrix with block32 with K=96 with SFA\_ID=01[](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-mma-scale-factor-a-block32-k96-dig2 "Permalink to this image")


![_images/tcgen05-mma-scale-factor-a-block32-k96-dig3.png](./ptx_files/tcgen05-mma-scale-factor-a-block32-k96-dig3.png)


Figure 236 Layout of scale factor A matrix with block32 with K=96 with SFA\_ID=10[](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-mma-scale-factor-a-block32-k96-dig3 "Permalink to this image")


![_images/tcgen05-mma-scale-factor-a-block32-k96-dig4.png](./ptx_files/tcgen05-mma-scale-factor-a-block32-k96-dig4.png)


Figure 237 Layout of scale factor A matrix with block32 with K=96 with SFA\_ID=11[](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-mma-scale-factor-a-block32-k96-dig4 "Permalink to this image")

For example, if *SFA\_ID* is 0, then all the green columns are selected to form the scale factor
matrix. Similarly, *SFA\_ID* values of 1, 2 and 3 would select the blue, yellow, and red columns,
respectively.
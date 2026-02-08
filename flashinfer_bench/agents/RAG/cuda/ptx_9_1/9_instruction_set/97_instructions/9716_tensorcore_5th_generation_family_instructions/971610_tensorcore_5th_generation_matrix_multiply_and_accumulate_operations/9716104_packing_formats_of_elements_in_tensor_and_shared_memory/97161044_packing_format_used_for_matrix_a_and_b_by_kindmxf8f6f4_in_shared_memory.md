# 9.7.16.10.4.4. Packing format used for matrix A and B by .kind::mxf8f6f4 in Shared Memory

###### 9.7.16.10.4.4. [Packing format used for matrix A and B by `.kind::mxf8f6f4` in Shared Memory](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-packing-formats-mxf8f6f4-smem)[](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-packing-formats-mxf8f6f4-smem "Permalink to this headline")

The 4-bit and 6-bit floating point elements in shared memory must be contiguously packed along
with padding as follows.

* 4-bit packing format as shown in [Figure 202](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-packing-formats-mxf8f6f4-smem-dig1)

  ![_images/tcgen05-packing-formats-mxf8f6f4-smem-dig1.png](./ptx_files/tcgen05-packing-formats-mxf8f6f4-smem-dig1.png)


  Figure 202 4-bit packing format[](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-packing-formats-mxf8f6f4-smem-dig1 "Permalink to this image")
* 6-bit packing format as shown in [Figure 203](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-packing-formats-mxf8f6f4-smem-dig2)

> ![_images/tcgen05-packing-formats-mxf8f6f4-smem-dig2.png](./ptx_files/tcgen05-packing-formats-mxf8f6f4-smem-dig2.png)
>
>
> Figure 203 6-bit packing format[](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-packing-formats-mxf8f6f4-smem-dig2 "Permalink to this image")
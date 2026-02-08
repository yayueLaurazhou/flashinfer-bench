# 9.7.16.10.4.3. Packing format used for matrix A by .kind::mxf8f6f4 in Tensor Memory

###### 9.7.16.10.4.3. [Packing format used for matrix A by `.kind::mxf8f6f4` in Tensor Memory](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-packing-formats-mxf8f6f4-tmem)[](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-packing-formats-mxf8f6f4-tmem "Permalink to this headline")

The individual 4-bit and the 6-bit floating point type elements must be packed in an 8-bit container
in Tensor memory as shown below. The 8-bit containers must be contiguously packed in a 32-bit Tensor
Memory word. For example, if the type of elements of the matrix `A` is 6 bits then 4 consecutive
`A` elements should be packed in one 32-bit Tensor Memory word.

* 4-bit packing format as shown in [Figure 199](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-packing-formats-mxf8f6f4-tmem-dig1)

  ![_images/tcgen05-packing-formats-mxf8f6f4-tmem-dig1.png](./ptx_files/tcgen05-packing-formats-mxf8f6f4-tmem-dig1.png)


  Figure 199 4-bit packing format with type E2M1[](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-packing-formats-mxf8f6f4-tmem-dig1 "Permalink to this image")
* 6-bit packing format

  + Type E3M2 as shown in [Figure 200](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-packing-formats-mxf8f6f4-tmem-dig2)

    ![_images/tcgen05-packing-formats-mxf8f6f4-tmem-dig2.png](./ptx_files/tcgen05-packing-formats-mxf8f6f4-tmem-dig2.png)


    Figure 200 6-bit packing format with type E3M2[](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-packing-formats-mxf8f6f4-tmem-dig2 "Permalink to this image")
  + Type E2M3 as shown in [Figure 201](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-packing-formats-mxf8f6f4-tmem-dig3)

    ![_images/tcgen05-packing-formats-mxf8f6f4-tmem-dig3.png](./ptx_files/tcgen05-packing-formats-mxf8f6f4-tmem-dig3.png)


    Figure 201 6-bit packing format with type E2M3[](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-packing-formats-mxf8f6f4-tmem-dig3 "Permalink to this image")
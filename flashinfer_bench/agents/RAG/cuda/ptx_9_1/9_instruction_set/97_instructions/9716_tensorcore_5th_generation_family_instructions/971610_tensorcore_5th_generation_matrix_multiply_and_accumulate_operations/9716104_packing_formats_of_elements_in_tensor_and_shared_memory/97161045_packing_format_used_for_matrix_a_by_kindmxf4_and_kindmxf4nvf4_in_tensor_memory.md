# 9.7.16.10.4.5. Packing format used for matrix A by .kind::mxf4 and .kind::mxf4nvf4 in Tensor Memory

###### 9.7.16.10.4.5. [Packing format used for matrix A by `.kind::mxf4` and `.kind::mxf4nvf4` in Tensor Memory](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-packing-formats-mxf4-tmem)[](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-packing-formats-mxf4-tmem "Permalink to this headline")

Two 4-bit floating point type elements must be packed in an 8-bit container in Tensor memory as
shown in [Figure 204](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-packing-formats-mxf4-tmem-dig1) for `mxf4`.

![_images/tcgen05-packing-formats-mxf4-tmem-dig1.png](./ptx_files/tcgen05-packing-formats-mxf4-tmem-dig1.png)


Figure 204 4-bit packing format with type E2M1[](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-packing-formats-mxf4-tmem-dig1 "Permalink to this image")
# 9.7.16.10.4.6. Packing format used for matrix A and B by .kind::mxf4 and .kind::mxf4nvf4 in Shared Memory

###### 9.7.16.10.4.6. [Packing format used for matrix A and B by `.kind::mxf4` and `.kind::mxf4nvf4` in Shared Memory](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-packing-formats-mxf4-smem)[](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-packing-formats-mxf4-smem "Permalink to this headline")

The packing format for 4-bit floating point elements in shared memory is to pack two 4-bit
elements in a 8-bit container, with no padding.
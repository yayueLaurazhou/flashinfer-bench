# 9.7.16.10.4.2. Packing format for matrix A and B

###### 9.7.16.10.4.2. [Packing format for matrix A and B](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-packing-formats-mat-a-b)[](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-packing-formats-mat-a-b "Permalink to this headline")

The 6-bit and 4-bit floating point types have different packing format requirements for
different MMA kinds in both Tensor memory and Shared memory. The requirements are as follows.
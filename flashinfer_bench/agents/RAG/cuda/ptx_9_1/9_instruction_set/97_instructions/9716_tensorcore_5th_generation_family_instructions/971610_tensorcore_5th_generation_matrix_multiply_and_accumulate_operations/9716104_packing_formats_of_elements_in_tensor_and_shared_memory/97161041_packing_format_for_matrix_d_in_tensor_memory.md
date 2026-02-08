# 9.7.16.10.4.1. Packing format for matrix D in Tensor Memory

###### 9.7.16.10.4.1. [Packing format for matrix D in Tensor Memory](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-packing-formats-mat-d)[](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-packing-formats-mat-d "Permalink to this headline")

The sub-word elements of matrix `D` are expected not to be packed within a 32-bit Tensor Memory word.
For example, if the type of elements of the matrix `D` is 16 bits then a Tensor Memory word
would contain a single 16-bit element in its lower 16 bits.
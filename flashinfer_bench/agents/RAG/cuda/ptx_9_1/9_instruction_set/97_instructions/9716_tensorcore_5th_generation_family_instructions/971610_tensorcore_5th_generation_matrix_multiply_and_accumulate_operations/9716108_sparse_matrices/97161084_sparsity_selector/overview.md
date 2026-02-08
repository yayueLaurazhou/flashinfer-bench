# 9.7.16.10.8.4. Sparsity selector

###### 9.7.16.10.8.4. [Sparsity selector](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-sparse-matrices-sparsity-selector)[](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-sparse-matrices-sparsity-selector "Permalink to this headline")

The value of the sparsity selector selects the sub-columns in the Tensor Memory
to form the sparsity metadata matrix, which is used with matrix `A` to form the
multiplicand matrix.

The following shows the sparse metadata matrix layout in Tensor Memory for various MMA variants:
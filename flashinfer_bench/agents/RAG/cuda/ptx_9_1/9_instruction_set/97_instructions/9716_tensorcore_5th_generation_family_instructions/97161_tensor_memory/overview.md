# 9.7.16.1. Tensor Memory

#### 9.7.16.1. [Tensor Memory](https://docs.nvidia.com/cuda/parallel-thread-execution/#tensor-memory)[](https://docs.nvidia.com/cuda/parallel-thread-execution/#tensor-memory "Permalink to this headline")

The 5th generation TensorCore has dedicated on-chip memory that is specialized for use by
TensorCore operations. This Tensor Memory is organized as a two-dimensional matrix where
the horizontal rows are called lanes and the vertical columns are called columns.

On architecture `sm_100a`/`sm_100f`, the 5th generation TensorCore’s Tensor Memory has a
two-dimensional structure of 512 columns and 128 rows per CTA, with each cell being 32-bits in size.

Restrictions on threads accessing the Tensor Memory via the load and store operations
are specified in [Access restrictions](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-tensor-memory-ld-st-access-restrictions).
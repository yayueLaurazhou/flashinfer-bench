# 9.7.16.1.2. Tensor Memory Allocation

##### 9.7.16.1.2. [Tensor Memory Allocation](https://docs.nvidia.com/cuda/parallel-thread-execution/#tensor-memory-allocation)[](https://docs.nvidia.com/cuda/parallel-thread-execution/#tensor-memory-allocation "Permalink to this headline")

The Tensor Memory is dynamically allocated. The Tensor Memory must be allocated by a single
warp in a CTA using the
[Tensor Memory Allocation and Management Instructions](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-memory-alloc-manage-instructions).

The allocation and deallocation of [Tensor Memory](https://docs.nvidia.com/cuda/parallel-thread-execution/#tensor-memory) is performed in terms of
columns. The unit of allocation is 32 columns and the number of columns being allocated must be
a power of 2. When a column is allocated, all 128 lanes of the column are allocated.

All of the Tensor Memory that was allocated in a kernel, must be explicitly deallocated
before the kernel exits.
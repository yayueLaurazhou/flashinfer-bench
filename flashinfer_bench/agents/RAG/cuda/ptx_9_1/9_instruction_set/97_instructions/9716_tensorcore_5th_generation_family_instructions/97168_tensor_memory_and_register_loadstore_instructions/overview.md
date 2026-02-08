# 9.7.16.8. Tensor Memory and Register Load/Store Instructions

#### 9.7.16.8. [Tensor Memory and Register Load/Store Instructions](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-tensor-memory-ld-st)[](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-tensor-memory-ld-st "Permalink to this headline")

The threads of the CTA can perform the loads and stores to the [Tensor Memory](https://docs.nvidia.com/cuda/parallel-thread-execution/#tensor-memory)
of the CTA and move data between registers and Tensor Memory. The loads and stores of data
can be performed in certain shapes as specified in the
[Matrix and Data Movement Shape](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-matrix-data-movement-shape) section.
# 9.7.16.10.8.5. Alignment restriction

###### 9.7.16.10.8.5. [Alignment restriction](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-sparse-matrices-alignment-restriction)[](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-sparse-matrices-alignment-restriction "Permalink to this headline")

The layouts which utilize only half the datapath lanes as specified in
[Data Path Layout Organization](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-data-path-layout-organization),
i.e. [Layout F](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-data-path-layout-f) and
[Layout C](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-data-path-layout-c), must use the same alignment
across matrices A, D and the sparsity metadata matrix.
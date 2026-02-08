# 9.7.16.2.2. Specifying Matrix Shape

##### 9.7.16.2.2. [Specifying Matrix Shape](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-specify-matrix-shape)[](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-specify-matrix-shape "Permalink to this headline")

*M* and *N* can be specified in the [Instruction descriptor](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-instruction-descriptor).

*K* cannot be explicitly specified but is implicitly determined by the MMA-kind
and the sparsity, as shown in the [Table 39](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-kind-shapes).
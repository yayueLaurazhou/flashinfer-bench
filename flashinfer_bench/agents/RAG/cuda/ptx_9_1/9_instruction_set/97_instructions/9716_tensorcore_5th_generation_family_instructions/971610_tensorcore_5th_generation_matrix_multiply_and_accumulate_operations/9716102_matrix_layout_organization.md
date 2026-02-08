# 9.7.16.10.2. Matrix Layout Organization

##### 9.7.16.10.2. [Matrix Layout Organization](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-matrix-layout-organization)[](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-matrix-layout-organization "Permalink to this headline")

[Table 53](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-matrices-majorness) describes the major-ness used for different matrices.

Table 53 Major-ness for different matrices[](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-matrices-majorness "Permalink to this table")





| Matrix | Residing in Memory | Default Major-ness |
| --- | --- | --- |
| D | Tensor Memory | Row-Major |
| A | Tensor Memory |
| Shared Memory | Depends on swizzling mode. Refer [Shared Memory Layout and Swizzling](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-shared-memory-layout-swizzling) |
| B | Shared Memory |
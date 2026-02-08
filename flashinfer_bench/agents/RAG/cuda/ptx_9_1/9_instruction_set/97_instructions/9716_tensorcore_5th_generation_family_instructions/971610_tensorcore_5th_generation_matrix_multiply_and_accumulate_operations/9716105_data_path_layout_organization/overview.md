# 9.7.16.10.5. Data Path Layout Organization

##### 9.7.16.10.5. [Data Path Layout Organization](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-data-path-layout-organization)[](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-data-path-layout-organization "Permalink to this headline")

Different MMA variants access the tensor memory with different layout organization.
The following table lists the various layouts:

| M | cta\_group | A-Sparsity | Is .ws mode | Datapath organization | Layout ID | Tensor Memory Datapath Lane Alignment |
| --- | --- | --- | --- | --- | --- | --- |
| 32 | ::1 | Either | Yes | 1x4 | [Layout G](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-data-path-layout-g) | 0 |
| 64 | ::1 | Either | Yes | 2x3 | [Layout E](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-data-path-layout-e) | 0 |
| 64 | ::1 | Either | No | 4x1 (1/2 datapath utilized) | [Layout F](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-data-path-layout-f) | 0 or 16 |
| 128 | ::1 | Either | Either | 4x1 | [Layout D](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-data-path-layout-d) | 0 |
| 128 | ::2 | Dense | N/A | 2x2 | [Layout B](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-data-path-layout-b) | 0 |
| 128 | ::2 | Sparse | N/A | 4x1 (1/2 datapath utilized) | [Layout C](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-data-path-layout-c) | 0 or 16 |
| 256 | ::2 | Either | N/A | 4x1 | [Layout A](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-data-path-layout-a) | 0 |

The layouts which utilize only half the datapath lanes, i.e.,
[Layout F](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-data-path-layout-f) and
[Layout C](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-data-path-layout-c), must use the same Tensor Memory
lane alignment across matrices `A`, `D` and the sparsity metadata matrix.

The following shows the warps that can access the Tensor Memory regions via
`tcgen05.ld` / `tcgen05.st` along with the addresses for various Tensor Memory Layouts.
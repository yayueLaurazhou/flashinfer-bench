# 9.7.15.5.1. Register Fragments and Shared Memory Matrix Layouts

##### 9.7.15.5.1. [Register Fragments and Shared Memory Matrix Layouts](https://docs.nvidia.com/cuda/parallel-thread-execution/#asynchronous-warpgroup-level-matrix-fragment)[](https://docs.nvidia.com/cuda/parallel-thread-execution/#asynchronous-warpgroup-level-matrix-fragment "Permalink to this headline")

The input matrix A of the warpgroup wide MMA operations can be either in registers or in the shared
memory. The input matrix B of the warpgroup wide MMA operations must be in the shared memory. This
section describes the layouts of register fragments and shared memory expected by the warpgroup MMA
instructions.

When the matrices are in shared memory, their starting addresses must be aligned to 16 bytes.
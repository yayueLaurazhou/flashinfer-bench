# 20.8.4. Features Accelerating Specialized Computations

### 20.8.4. Features Accelerating Specialized Computations[](#features-accelerating-specialized-computations "Permalink to this headline")

The NVIDIA Hopper GPU architecture includes features to accelerate matrix multiply-accumulate (MMA) computations with:

* asynchronous execution of MMA instructions
* MMA instructions acting on large matrices spanning a warp-group
* dynamic reassignment of register capacity among warp-groups to support even larger matrices, and
* operand matrices accessed directly from shared memory

See the [PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#instruction-set) for more details.

This feature set is only available within the CUDA compilation toolchain through inline PTX.

It is strongly recommended that applications utilize this complex feature set through CUDA-X libraries such as cuBLAS, cuDNN, or cuFFT.

It is strongly recommended that device kernels utilize this complex feature set through [CUTLASS](https://github.com/NVIDIA/cutlass), a collection of CUDA C++ template abstractions for implementing high-performance matrix-multiplication (GEMM) and related computations at all levels and scales within CUDA.
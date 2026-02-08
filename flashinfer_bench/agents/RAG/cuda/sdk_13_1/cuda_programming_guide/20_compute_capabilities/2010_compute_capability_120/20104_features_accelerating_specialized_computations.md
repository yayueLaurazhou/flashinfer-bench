# 20.10.4. Features Accelerating Specialized Computations

### 20.10.4. Features Accelerating Specialized Computations[](#features-accelerating-specialized-computations-12-0 "Permalink to this headline")

The NVIDIA Blackwell GPU architecture extends features to accelerate matrix multiply-accumulate (MMA) from the NVIDIA Hopper GPU architecture.

See the [PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#instruction-set) for more details.

This feature set is only available within the CUDA compilation toolchain through inline PTX.

It is strongly recommended that applications utilize this complex feature set through CUDA-X libraries such as cuBLAS, cuDNN, or cuFFT.

It is strongly recommended that device kernels utilize this complex feature set through [CUTLASS](https://github.com/NVIDIA/cutlass), a collection of CUDA C++ template abstractions for implementing high-performance matrix-multiplication (GEMM) and related computations at all levels and scales within CUDA.

[27](#id399)
:   above 48 KB requires dynamic shared memory

[28](#id410)
:   2 FP64 cores for double-precision arithmetic operations for devices of compute capabilities 7.5
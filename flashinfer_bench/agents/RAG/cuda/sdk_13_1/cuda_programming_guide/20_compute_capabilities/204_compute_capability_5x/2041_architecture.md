# 20.4.1. Architecture

### 20.4.1. Architecture[](#architecture "Permalink to this headline")

An SM consists of:

* 128 CUDA cores for arithmetic operations (see [CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#arithmetic-instructions) for throughputs of arithmetic operations),
* 32 special function units for single-precision floating-point transcendental functions,
* 4 warp schedulers.

When an SM is given warps to execute, it first distributes them among the four schedulers. Then, at every instruction issue time, each scheduler issues one instruction for one of its assigned warps that is ready to execute, if any.

An SM has:

* a read-only constant cache that is shared by all functional units and speeds up reads from the constant memory space, which resides in device memory,
* a unified L1/texture cache of 24 KB used to cache reads from global memory,
* 64 KB of shared memory for devices of compute capability 5.0 or 96 KB of shared memory for devices of compute capability 5.2.

The unified L1/texture cache is also used by the texture unit that implements the various addressing modes and data filtering mentioned in [Texture and Surface Memory](#texture-and-surface-memory).

There is also an L2 cache shared by all SMs that is used to cache accesses to local or global memory, including temporary register spills. Applications may query the L2 cache size by checking the `l2CacheSize` device property (see [Device Enumeration](#device-enumeration)).

The cache behavior (e.g., whether reads are cached in both the unified L1/texture cache and L2 or in L2 only) can be partially configured on a per-access basis using modifiers to the load instruction.
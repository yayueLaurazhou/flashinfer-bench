# 20.9.1. Architecture

### 20.9.1. Architecture[](#architecture-10-0 "Permalink to this headline")

A Streaming Multiprocessor (SM) consists of:

* 128 FP32 cores for single-precision arithmetic operations,
* 64 FP64 cores for double-precision arithmetic operations,
* 64 INT32 cores for integer math,
* 4 mixed-precision fifth-generation Tensor Cores supporting `FP8` input type in either `E4M3` or `E5M2` for exponent (E) and mantissa (M), half-precision (fp16), `__nv_bfloat16`, `tf32`, INT8 and double precision (fp64) matrix arithmetic (see [Warp Matrix Functions](#wmma) for details) with sparsity support,
* 16 special function units for single-precision floating-point transcendental functions,
* 4 warp schedulers.

An SM statically distributes its warps among its schedulers. Then, at every instruction issue time, each scheduler issues one instruction for one of its assigned warps that is ready to execute, if any.

An SM has:

* a read-only constant cache that is shared by all functional units and speeds up reads from the constant memory space, which resides in device memory,
* a unified data cache and shared memory with a total size of 256 KB for devices of compute capability 10.0

Shared memory is partitioned out of the unified data cache, and can be configured to various sizes (see [Shared Memory](#shared-memory-10-0)). The remaining data cache serves as an L1 cache and is also used by the texture unit that implements the various addressing and data filtering modes mentioned in [Texture and Surface Memory](#texture-and-surface-memory).
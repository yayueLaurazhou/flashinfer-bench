# 20.6.1. Architecture

### 20.6.1. Architecture[](#architecture-7-x "Permalink to this headline")

An SM consists of:

* 64 FP32 cores for single-precision arithmetic operations,
* 32 FP64 cores for double-precision arithmetic operations,[28](#fn35)
* 64 INT32 cores for integer math,
* 8 mixed-precision Tensor Cores for deep learning matrix arithmetic
* 16 special function units for single-precision floating-point transcendental functions,
* 4 warp schedulers.

An SM statically distributes its warps among its schedulers. Then, at every instruction issue time, each scheduler issues one instruction for one of its assigned warps that is ready to execute, if any.

An SM has:

* a read-only constant cache that is shared by all functional units and speeds up reads from the constant memory space, which resides in device memory,
* a unified data cache and shared memory with a total size of 128 KB (*Volta*) or 96 KB (*Turing*).

Shared memory is partitioned out of unified data cache, and can be configured to various sizes (See [Shared Memory](#shared-memory-7-x).) The remaining data cache serves as an L1 cache and is also used by the texture unit that implements the various addressing and data filtering modes mentioned in [Texture and Surface Memory](#texture-and-surface-memory).
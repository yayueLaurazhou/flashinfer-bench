# 20.5.1. Architecture

### 20.5.1. Architecture[](#architecture-6-x "Permalink to this headline")

An SM consists of:

* 64 (compute capability 6.0) or 128 (6.1 and 6.2) CUDA cores for arithmetic operations,
* 16 (6.0) or 32 (6.1 and 6.2) special function units for single-precision floating-point transcendental functions,
* 2 (6.0) or 4 (6.1 and 6.2) warp schedulers.

When an SM is given warps to execute, it first distributes them among its schedulers. Then, at every instruction issue time, each scheduler issues one instruction for one of its assigned warps that is ready to execute, if any.

An SM has:

* a read-only constant cache that is shared by all functional units and speeds up reads from the constant memory space, which resides in device memory,
* a unified L1/texture cache for reads from global memory of size 24 KB (6.0 and 6.2) or 48 KB (6.1),
* a shared memory of size 64 KB (6.0 and 6.2) or 96 KB (6.1).

The unified L1/texture cache is also used by the texture unit that implements the various addressing modes and data filtering mentioned in [Texture and Surface Memory](#texture-and-surface-memory).

There is also an L2 cache shared by all SMs that is used to cache accesses to local or global memory, including temporary register spills. Applications may query the L2 cache size by checking the `l2CacheSize` device property (see [Device Enumeration](#device-enumeration)).

The cache behavior (for example, whether reads are cached in both the unified L1/texture cache and L2 or in L2 only) can be partially configured on a per-access basis using modifiers to the load instruction.
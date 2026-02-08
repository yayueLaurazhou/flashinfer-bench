# 10.29.3. TMA Swizzle

### 10.29.3. TMA Swizzle[](#tma-swizzle "Permalink to this headline")

By default, the TMA engine loads data to shared memory in the same order as it is laid out in global memory. However, this
layout may not be optimal for certain shared memory access patterns, as it could cause shared memory bank conflicts. To
improve performance and reduce bank conflicts, we can change the shared memory layout by applying a ‘swizzle pattern’.

Shared memory has 32 banks that are organized such that successive 32-bit words map to successive banks. Each bank has a
bandwidth of 32 bits per clock cycle. When loading and storing shared memory, bank conflicts arise if the same bank is
used multiple times within a transaction, resulting in reduced bandwidth. See [Shared Memory](#shared-memory-5-x), bank conflicts.

To ensure that data is laid out in shared memory in such a way that user code can avoid shared memory bank conflicts,
the TMA engine can be instructed to ‘swizzle’ the data before storing it in shared memory and ‘unswizzle’ it when copying
the data back from shared memory to global memory. The tensor map encodes the ‘swizzle mode’ indicating which swizzle pattern is used.
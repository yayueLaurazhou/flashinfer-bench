# 10.29.3.2. The Swizzle Modes

#### 10.29.3.2. The Swizzle Modes[](#the-swizzle-modes "Permalink to this headline")

As previously mentioned, there are four swizzle modes. The following tables show the different swizzle patterns, including the relation of the new
shared memory indices. The tables define the mapping of the 16-byte chunks along the 128 bytes to eight subgroups of four banks.

[![An Overview of TMA Swizzle Patterns](_images/swizzle-pattern.png)](_images/swizzle-pattern.png)


Figure 29 An Overview of TMA Swizzle Patterns[](#id470 "Permalink to this image")

**Considerations.** When applying a TMA swizzle pattern, it is crucial to adhere to specific memory requirements:

* **Global memory alignment:**
  Global memory must be aligned to 128 bytes.
* **Shared memory alignment:**
  For simplicity shared memory should be aligned according to the number of bytes after which the swizzle pattern repeats. When the shared memory buffer is not aligned by the number of bytes by which the swizzle pattern repeats itself, there is an offset between the swizzle pattern and the shared memory.
  See [comment](#swizzle-pattern-pointer-offset-computation), below.
* **Inner dimension:**
  The inner dimension of the shared memory block must meet the size requirements specified in [Table 12](#table-swizzle-pattern-properties-and-requirements). If these
  requirements are not met, the instruction is considered invalid. Additionally, if the swizzle width exceeds the inner dimension,
  ensure that the shared memory is allocated to accommodate the full swizzle width.
* **Granularity:**
  The granularity of swizzle mapping is fixed at 16 bytes. This means that data is organized and accessed in chunks
  of 16 bytes, which must be considered when planning memory layout and access patterns.

**Swizzle Pattern Pointer Offset Computation**. Here, we describe how to determine the offset between the swizzle pattern and the shared memory, when the shared memory buffer is not aligned by the number of bytes by which the swizzle pattern repeats itself.
When using TMA, the shared memory is required to be aligned to 128 bytes. To find how many times the shared memory buffer relative to the swizzle pattern is shifted by that, apply the corresponding offset formula.

Table 11 Swizzle Pattern Pointer Offset Formula and Index Relation[](#table-swizzle-pattern-offset "Permalink to this table")

| Swizzle Mode | Offset Formula | Index Relation |
| --- | --- | --- |
| CU\_TENSOR\_MAP\_SWIZZLE\_128B | `(reinterpret_cast <uintptr_t>(smem_ptr)/128)%8` | `smem[y][x] <-> smem[y][((y+offset)%8)^x]` |
| CU\_TENSOR\_MAP\_SWIZZLE\_64B | `(reinterpret_cast <uintptr_t>(smem_ptr)/128)%4` | `smem[y][x] <-> smem[y][((y+offset)%4)^x]` |
| CU\_TENSOR\_MAP\_SWIZZLE\_32B | `(reinterpret_cast <uintptr_t>(smem_ptr)/128)%2` | `smem[y][x] <-> smem[y][((y+offset)%2)^x]` |

In [Figure 29](#figure-swizzle-overview), this offset represents the initial row offset, thus, in the swizzle index calculation, it is added to the row index `y`.
The following snippet shows how to access the swizzled shared memory in the `CU_TENSOR_MAP_SWIZZLE_128B` mode.

```
data_t* smem_ptr = &smem[0][0];
int offset = (reinterpret_cast<uintptr_t>(smem_ptr)/128)%8;
smem[y][((y+offset)%8)^x] = ...
```

**Summary.** The following [Table 12](#table-swizzle-pattern-properties-and-requirements) summarizes the requirements and properties of the different swizzle patterns for Compute Capability 9.

Table 12 Requirements and properties of the different swizzle patterns for Compute Capability 9[](#table-swizzle-pattern-properties-and-requirements "Permalink to this table")

| Pattern | Swizzle width | Shared box’s inner dimension | Repeats after | Shared memory alignment | Global memory alignment |
| --- | --- | --- | --- | --- | --- |
| CU\_TENSOR\_MAP\_SWIZZLE\_128B | 128 bytes | <=128 bytes | 1024 bytes | 128 bytes | 128 bytes |
| CU\_TENSOR\_MAP\_SWIZZLE\_64B | 64 bytes | <=64 bytes | 512 bytes | 128 bytes | 128 bytes |
| CU\_TENSOR\_MAP\_SWIZZLE\_32B | 32 bytes | <=32 bytes | 256 bytes | 128 bytes | 128 bytes |
| CU\_TENSOR\_MAP\_SWIZZLE\_NONE (default) |  |  |  | 128 bytes | 16 bytes |
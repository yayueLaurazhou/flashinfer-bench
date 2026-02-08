# asynchronous-warpgroup-level-matrix-shared-memory-layout-matrix-descriptor

###### 9.7.15.5.1.2.2. [Matrix Descriptor Format](https://docs.nvidia.com/cuda/parallel-thread-execution/#asynchronous-warpgroup-level-matrix-shared-memory-layout-matrix-descriptor)[](https://docs.nvidia.com/cuda/parallel-thread-execution/#asynchronous-warpgroup-level-matrix-shared-memory-layout-matrix-descriptor "Permalink to this headline")

Matrix descriptor specifies the properties of the matrix in shared memory that is a multiplicand in
the matrix multiply and accumulate operation. It is a 64-bit value contained in a register with the
following layout:

| Bit-field | Size in bits | Description |
| --- | --- | --- |
| 13–0 | 14 | matrix-descriptor-encode(Matrix start address) |
| 29–16 | 14 | matrix-descriptor-encode ([Leading dimension byte offset](https://docs.nvidia.com/cuda/parallel-thread-execution/#asynchronous-warpgroup-level-leading-dimension-byte-offset)) |
| 45–32 | 14 | matrix-descriptor-encode ([Stride dimension byte offset](https://docs.nvidia.com/cuda/parallel-thread-execution/#asynchronous-warpgroup-level-stride-dimension-byte-offset)) |
| 51–49 | 3 | Matrix base offset. This is valid for all swizzling modes except the no-swizzle mode. |
| 63–62 | 2 | Specifies the swizzling mode to be used:   * 0: No swizzle * 1: 128-Byte swizzle * 2: 64-Byte swizzle * 3: 32-Byte swizzle |

where

```
matrix-descriptor-encode(x) = (x & 0x3FFFF) >> 4
```

Copy to clipboard

The value of base offset is 0 when the repeating pattern of the specified swizzling mode starts as
per the below table:

> | Swizzling mode | Starting address of the repeating pattern |
> | --- | --- |
> | 128-Byte swizzle | 1024-Byte boundary |
> | 64-Byte swizzle | 512-Byte boundary |
> | 32-Byte swizzle | 256-Byte boundary |

Otherwise, the base offset must be a non-zero value, computed using the following formula:

```
base offset = (pattern start addr >> 0x7) & 0x7
```

Copy to clipboard
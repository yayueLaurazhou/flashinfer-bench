# 9.7.1.2. Integer Arithmetic Instructions: sub

#### 9.7.1.2. [Integer Arithmetic Instructions: `sub`](https://docs.nvidia.com/cuda/parallel-thread-execution/#integer-arithmetic-instructions-sub)[](https://docs.nvidia.com/cuda/parallel-thread-execution/#integer-arithmetic-instructions-sub "Permalink to this headline")

`sub`

Subtract one value from another.

Syntax

```
sub.type       d, a, b;
sub{.sat}.s32  d, a, b;     // .sat applies only to .s32

.type = { .u16, .u32, .u64,
          .s16, .s32, .s64 };
```

Copy to clipboard

Description

Performs subtraction and writes the resulting value into a destination register.

Semantics

```
d = a - b;
```

Copy to clipboard

Notes

Saturation modifier:

`.sat`
:   limits result to `MININT..MAXINT` (no overflow) for the size of the operation. Applies only to
    `.s32` type.

PTX ISA Notes

Introduced in PTX ISA version 1.0.

Target ISA Notes

Supported on all target architectures.

Examples

```
sub.s32 c,a,b;
```

Copy to clipboard
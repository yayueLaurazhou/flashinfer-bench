# 9.7.1.9. Integer Arithmetic Instructions: rem

#### 9.7.1.9. [Integer Arithmetic Instructions: `rem`](https://docs.nvidia.com/cuda/parallel-thread-execution/#integer-arithmetic-instructions-rem)[](https://docs.nvidia.com/cuda/parallel-thread-execution/#integer-arithmetic-instructions-rem "Permalink to this headline")

`rem`

The remainder of integer division.

Syntax

```
rem.type  d, a, b;

.type = { .u16, .u32, .u64,
          .s16, .s32, .s64 };
```

Copy to clipboard

Description

Divides `a` by `b`, store the remainder in `d`.

Semantics

```
d = a % b;
```

Copy to clipboard

Notes

The behavior for negative numbers is machine-dependent and depends on whether divide rounds towards
zero or negative infinity.

Division by zero yields an unspecified, machine-specific value.

PTX ISA Notes

Introduced in PTX ISA version 1.0.

Target ISA Notes

Supported on all target architectures.

Examples

```
rem.s32  x,x,8;    // x = x%8;
```

Copy to clipboard
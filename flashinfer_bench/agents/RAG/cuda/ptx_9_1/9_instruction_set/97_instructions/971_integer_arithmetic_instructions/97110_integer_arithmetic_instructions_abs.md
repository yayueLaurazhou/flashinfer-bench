# 9.7.1.10. Integer Arithmetic Instructions: abs

#### 9.7.1.10. [Integer Arithmetic Instructions: `abs`](https://docs.nvidia.com/cuda/parallel-thread-execution/#integer-arithmetic-instructions-abs)[](https://docs.nvidia.com/cuda/parallel-thread-execution/#integer-arithmetic-instructions-abs "Permalink to this headline")

`abs`

Absolute value.

Syntax

```
abs.type  d, a;

.type = { .s16, .s32, .s64 };
```

Copy to clipboard

Description

Take the absolute value of `a` and store it in `d`.

Semantics

```
d = |a|;
```

Copy to clipboard

Notes

Only for signed integers.

PTX ISA Notes

Introduced in PTX ISA version 1.0.

Target ISA Notes

Supported on all target architectures.

Examples

```
abs.s32  r0,a;
```

Copy to clipboard
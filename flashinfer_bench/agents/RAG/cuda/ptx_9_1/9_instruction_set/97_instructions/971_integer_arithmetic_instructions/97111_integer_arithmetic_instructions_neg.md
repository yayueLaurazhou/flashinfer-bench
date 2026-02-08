# 9.7.1.11. Integer Arithmetic Instructions: neg

#### 9.7.1.11. [Integer Arithmetic Instructions: `neg`](https://docs.nvidia.com/cuda/parallel-thread-execution/#integer-arithmetic-instructions-neg)[](https://docs.nvidia.com/cuda/parallel-thread-execution/#integer-arithmetic-instructions-neg "Permalink to this headline")

`neg`

Arithmetic negate.

Syntax

```
neg.type  d, a;

.type = { .s16, .s32, .s64 };
```

Copy to clipboard

Description

Negate the sign of **a** and store the result in **d**.

Semantics

```
d = -a;
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
neg.s32  r0,a;
```

Copy to clipboard
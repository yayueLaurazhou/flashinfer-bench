# 9.7.1.8. Integer Arithmetic Instructions: div

#### 9.7.1.8. [Integer Arithmetic Instructions: `div`](https://docs.nvidia.com/cuda/parallel-thread-execution/#integer-arithmetic-instructions-div)[](https://docs.nvidia.com/cuda/parallel-thread-execution/#integer-arithmetic-instructions-div "Permalink to this headline")

`div`

Divide one value by another.

Syntax

```
div.type  d, a, b;

.type = { .u16, .u32, .u64,
          .s16, .s32, .s64 };
```

Copy to clipboard

Description

Divides `a` by `b`, stores result in `d`.

Semantics

```
d = a / b;
```

Copy to clipboard

Notes

Division by zero yields an unspecified, machine-specific value.

PTX ISA Notes

Introduced in PTX ISA version 1.0.

Target ISA Notes

Supported on all target architectures.

Examples

```
div.s32  b,n,i;
```

Copy to clipboard
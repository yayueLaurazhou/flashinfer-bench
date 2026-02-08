# 9.7.1.7. Integer Arithmetic Instructions: sad

#### 9.7.1.7. [Integer Arithmetic Instructions: `sad`](https://docs.nvidia.com/cuda/parallel-thread-execution/#integer-arithmetic-instructions-sad)[](https://docs.nvidia.com/cuda/parallel-thread-execution/#integer-arithmetic-instructions-sad "Permalink to this headline")

`sad`

Sum of absolute differences.

Syntax

```
sad.type  d, a, b, c;

.type = { .u16, .u32, .u64,
          .s16, .s32, .s64 };
```

Copy to clipboard

Description

Adds the absolute value of `a-b` to `c` and writes the resulting value into `d`.

Semantics

```
d = c + ((a<b) ? b-a : a-b);
```

Copy to clipboard

PTX ISA Notes

Introduced in PTX ISA version 1.0.

Target ISA Notes

Supported on all target architectures.

Examples

```
sad.s32  d,a,b,c;
sad.u32  d,a,b,d;  // running sum
```

Copy to clipboard
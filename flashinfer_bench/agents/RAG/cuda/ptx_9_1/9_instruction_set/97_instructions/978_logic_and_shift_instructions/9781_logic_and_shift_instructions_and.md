# 9.7.8.1. Logic and Shift Instructions: and

#### 9.7.8.1. [Logic and Shift Instructions: `and`](https://docs.nvidia.com/cuda/parallel-thread-execution/#logic-and-shift-instructions-and)[](https://docs.nvidia.com/cuda/parallel-thread-execution/#logic-and-shift-instructions-and "Permalink to this headline")

`and`

Bitwise AND.

Syntax

```
and.type d, a, b;

.type = { .pred, .b16, .b32, .b64 };
```

Copy to clipboard

Description

Compute the bit-wise and operation for the bits in `a` and `b`.

Semantics

```
d = a & b;
```

Copy to clipboard

Notes

The size of the operands must match, but not necessarily the type.

Allowed types include predicate registers.

PTX ISA Notes

Introduced in PTX ISA version 1.0.

Target ISA Notes

Supported on all target architectures.

Examples

```
and.b32  x,q,r;
and.b32  sign,fpvalue,0x80000000;
```

Copy to clipboard
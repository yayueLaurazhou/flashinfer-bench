# 9.7.8.2. Logic and Shift Instructions: or

#### 9.7.8.2. [Logic and Shift Instructions: `or`](https://docs.nvidia.com/cuda/parallel-thread-execution/#logic-and-shift-instructions-or)[](https://docs.nvidia.com/cuda/parallel-thread-execution/#logic-and-shift-instructions-or "Permalink to this headline")

`or`

Biwise OR.

Syntax

```
or.type d, a, b;

.type = { .pred, .b16, .b32, .b64 };
```

Copy to clipboard

Description

Compute the bit-wise or operation for the bits in `a` and `b`.

Semantics

```
d = a | b;
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
or.b32  mask mask,0x00010001
or.pred  p,q,r;
```

Copy to clipboard
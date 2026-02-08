# 9.7.8.4. Logic and Shift Instructions: not

#### 9.7.8.4. [Logic and Shift Instructions: `not`](https://docs.nvidia.com/cuda/parallel-thread-execution/#logic-and-shift-instructions-not)[](https://docs.nvidia.com/cuda/parallel-thread-execution/#logic-and-shift-instructions-not "Permalink to this headline")

`not`

Bitwise negation; one’s complement.

Syntax

```
not.type d, a;

.type = { .pred, .b16, .b32, .b64 };
```

Copy to clipboard

Description

Invert the bits in `a`.

Semantics

```
d = ~a;
```

Copy to clipboard

Notes

The size of the operands must match, but not necessarily the type.

Allowed types include predicates.

PTX ISA Notes

Introduced in PTX ISA version 1.0.

Target ISA Notes

Supported on all target architectures.

Examples

```
not.b32  mask,mask;
not.pred  p,q;
```

Copy to clipboard
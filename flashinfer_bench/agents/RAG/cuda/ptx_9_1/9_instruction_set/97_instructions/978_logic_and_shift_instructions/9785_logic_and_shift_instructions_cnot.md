# 9.7.8.5. Logic and Shift Instructions: cnot

#### 9.7.8.5. [Logic and Shift Instructions: `cnot`](https://docs.nvidia.com/cuda/parallel-thread-execution/#logic-and-shift-instructions-cnot)[](https://docs.nvidia.com/cuda/parallel-thread-execution/#logic-and-shift-instructions-cnot "Permalink to this headline")

`cnot`

C/C++ style logical negation.

Syntax

```
cnot.type d, a;

.type = { .b16, .b32, .b64 };
```

Copy to clipboard

Description

Compute the logical negation using C/C++ semantics.

Semantics

```
d = (a==0) ? 1 : 0;
```

Copy to clipboard

Notes

The size of the operands must match, but not necessarily the type.

PTX ISA Notes

Introduced in PTX ISA version 1.0.

Target ISA Notes

Supported on all target architectures.

Examples

```
cnot.b32 d,a;
```

Copy to clipboard
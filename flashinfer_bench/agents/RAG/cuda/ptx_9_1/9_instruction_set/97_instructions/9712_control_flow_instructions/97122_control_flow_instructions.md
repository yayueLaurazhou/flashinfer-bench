# 9.7.12.2. Control Flow Instructions: @

#### 9.7.12.2. [Control Flow Instructions: `@`](https://docs.nvidia.com/cuda/parallel-thread-execution/#control-flow-instructions-at)[](https://docs.nvidia.com/cuda/parallel-thread-execution/#control-flow-instructions-at "Permalink to this headline")

`@`

Predicated execution.

Syntax

```
@{!}p    instruction;
```

Copy to clipboard

Description

Execute an instruction or instruction block for threads that have the guard predicate
`True`. Threads with a `False` guard predicate do nothing.

Semantics

If `{!}p` then instruction

PTX ISA Notes

Introduced in PTX ISA version 1.0.

Target ISA Notes

Supported on all target architectures.

Examples

```
    setp.eq.f32  p,y,0;     // is y zero?
@!p div.f32      ratio,x,y  // avoid division by zero

@q  bra L23;                // conditional branch
```

Copy to clipboard
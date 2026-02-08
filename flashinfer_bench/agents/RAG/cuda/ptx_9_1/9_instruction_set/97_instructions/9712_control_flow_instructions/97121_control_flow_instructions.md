# 9.7.12.1. Control Flow Instructions: {}

#### 9.7.12.1. [Control Flow Instructions: `{}`](https://docs.nvidia.com/cuda/parallel-thread-execution/#control-flow-instructions-curly-braces)[](https://docs.nvidia.com/cuda/parallel-thread-execution/#control-flow-instructions-curly-braces "Permalink to this headline")

`{}`

Instruction grouping.

Syntax

```
{ instructionList }
```

Copy to clipboard

Description

The curly braces create a group of instructions, used primarily for defining a function body. The
curly braces also provide a mechanism for determining the scope of a variable: any variable declared
within a scope is not available outside the scope.

PTX ISA Notes

Introduced in PTX ISA version 1.0.

Target ISA Notes

Supported on all target architectures.

Examples

```
{ add.s32  a,b,c; mov.s32  d,a; }
```

Copy to clipboard
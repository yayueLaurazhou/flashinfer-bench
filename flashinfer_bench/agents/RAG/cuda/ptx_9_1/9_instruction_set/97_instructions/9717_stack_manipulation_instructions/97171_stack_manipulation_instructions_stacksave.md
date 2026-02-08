# 9.7.17.1. Stack Manipulation Instructions: stacksave

#### 9.7.17.1. [Stack Manipulation Instructions: `stacksave`](https://docs.nvidia.com/cuda/parallel-thread-execution/#stack-manipulation-instructions-stacksave)[](https://docs.nvidia.com/cuda/parallel-thread-execution/#stack-manipulation-instructions-stacksave "Permalink to this headline")

`stacksave`

Save the value of stack pointer into a register.

Syntax

```
stacksave.type  d;

.type = { .u32, .u64 };
```

Copy to clipboard

Description

Copies the current value of stack pointer into the destination register `d`. Pointer returned by
`stacksave` can be used in a subsequent `stackrestore` instruction to restore the stack
pointer. If `d` is modified prior to use in `stackrestore` instruction, it may corrupt data in
the stack.

Destination operand `d` has the same type as the instruction type.

Semantics

```
d = stackptr;
```

Copy to clipboard

PTX ISA Notes

Introduced in PTX ISA version 7.3.

Target ISA Notes

`stacksave` requires `sm_52` or higher.

Examples

```
.reg .u32 rd;
stacksave.u32 rd;

.reg .u64 rd1;
stacksave.u64 rd1;
```

Copy to clipboard
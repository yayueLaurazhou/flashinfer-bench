# 9.7.17.2. Stack Manipulation Instructions: stackrestore

#### 9.7.17.2. [Stack Manipulation Instructions: `stackrestore`](https://docs.nvidia.com/cuda/parallel-thread-execution/#stack-manipulation-instructions-stackrestore)[](https://docs.nvidia.com/cuda/parallel-thread-execution/#stack-manipulation-instructions-stackrestore "Permalink to this headline")

`stackrestore`

Update the stack pointer with a new value.

Syntax

```
stackrestore.type  a;

.type = { .u32, .u64 };
```

Copy to clipboard

Description

Sets the current stack pointer to source register `a`.

When `stackrestore` is used with operand `a` written by a prior `stacksave` instruction, it
will effectively restore the state of stack as it was before `stacksave` was executed. Note that
if `stackrestore` is used with an arbitrary value of `a`, it may cause corruption of stack
pointer. This implies that the correct use of this feature requires that `stackrestore.type a` is
used after `stacksave.type a` without redefining the value of `a` between them.

Operand `a` has the same type as the instruction type.

Semantics

```
stackptr = a;
```

Copy to clipboard

PTX ISA Notes

Introduced in PTX ISA version 7.3.

Target ISA Notes

`stackrestore` requires `sm_52` or higher.

Examples

```
.reg .u32 ra;
stacksave.u32 ra;
// Code that may modify stack pointer
...
stackrestore.u32 ra;
```

Copy to clipboard
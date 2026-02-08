# 10.21. Special Registers: %lanemask_ge

## 10.21. [Special Registers: `%lanemask_ge`](https://docs.nvidia.com/cuda/parallel-thread-execution/#special-registers-lanemask-ge)[](https://docs.nvidia.com/cuda/parallel-thread-execution/#special-registers-lanemask-ge "Permalink to this headline")

`%lanemask_ge`

32-bit mask with bits set in positions greater than or equal to the thread’s lane number in the warp.

Syntax (predefined)

```
.sreg .u32 %lanemask_ge;
```

Copy to clipboard

Description

A predefined, read-only special register initialized with a 32-bit mask with bits set in positions
greater than or equal to the thread’s lane number in the warp.

PTX ISA Notes

Introduced in PTX ISA version 2.0.

Target ISA Notes

`%lanemask_ge` requires `sm_20` or higher.

Examples

```
mov.u32     %r, %lanemask_ge;
```

Copy to clipboard
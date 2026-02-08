# 10.3. Special Registers: %laneid

## 10.3. [Special Registers: `%laneid`](https://docs.nvidia.com/cuda/parallel-thread-execution/#special-registers-laneid)[](https://docs.nvidia.com/cuda/parallel-thread-execution/#special-registers-laneid "Permalink to this headline")

`%laneid`

Lane Identifier.

Syntax (predefined)

```
.sreg .u32 %laneid;
```

Copy to clipboard

Description

A predefined, read-only special register that returns the thread’s lane within the warp. The lane
identifier ranges from zero to `WARP_SZ-1`.

PTX ISA Notes

Introduced in PTX ISA version 1.3.

Target ISA Notes

Supported on all target architectures.

Examples

```
mov.u32  %r, %laneid;
```

Copy to clipboard
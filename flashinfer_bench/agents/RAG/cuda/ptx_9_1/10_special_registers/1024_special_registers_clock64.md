# 10.24. Special Registers: %clock64

## 10.24. [Special Registers: `%clock64`](https://docs.nvidia.com/cuda/parallel-thread-execution/#special-registers-clock64)[](https://docs.nvidia.com/cuda/parallel-thread-execution/#special-registers-clock64 "Permalink to this headline")

`%clock64`

A predefined, read-only 64-bit unsigned cycle counter.

Syntax (predefined)

```
.sreg .u64 %clock64;
```

Copy to clipboard

Description

Special register `%clock64` is an unsigned 64-bit read-only cycle counter that wraps silently.

Notes

The lower 32-bits of `%clock64` are identical to `%clock`.

The upper 32-bits of `%clock64` are identical to `%clock_hi`.

PTX ISA Notes

Introduced in PTX ISA version 2.0.

Target ISA Notes

`%clock64` requires `sm_20` or higher.

Examples

```
mov.u64  r1,%clock64;
```

Copy to clipboard
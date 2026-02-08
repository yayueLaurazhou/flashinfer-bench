# 10.32. Special Registers: %dynamic_smem_size

## 10.32. [Special Registers: `%dynamic_smem_size`](https://docs.nvidia.com/cuda/parallel-thread-execution/#special-registers-dynamic-smem-size)[](https://docs.nvidia.com/cuda/parallel-thread-execution/#special-registers-dynamic-smem-size "Permalink to this headline")

`%dynamic_smem_size`

Size of shared memory allocated dynamically at kernel launch.

Syntax (predefined)

```
.sreg .u32 %dynamic_smem_size;
```

Copy to clipboard

Description

Size of shared memory allocated dynamically at kernel launch.

A predefined, read-only special register initialized with size of shared memory allocated dynamically for the CTA of a kernel at launch time.

PTX ISA Notes

Introduced in PTX ISA version 4.1.

Target ISA Notes

Requires `sm_20` or higher.

Examples

```
mov.u32  %r, %dynamic_smem_size;
```

Copy to clipboard
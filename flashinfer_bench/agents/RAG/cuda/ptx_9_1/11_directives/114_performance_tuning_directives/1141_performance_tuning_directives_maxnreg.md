# 11.4.1. Performance-Tuning Directives: .maxnreg

### 11.4.1. [Performance-Tuning Directives: `.maxnreg`](https://docs.nvidia.com/cuda/parallel-thread-execution/#performance-tuning-directives-maxnreg)[](https://docs.nvidia.com/cuda/parallel-thread-execution/#performance-tuning-directives-maxnreg "Permalink to this headline")

`.maxnreg`

Maximum number of registers that can be allocated per thread.

Syntax

```
.maxnreg n
```

Copy to clipboard

Description

Declare the maximum number of registers per thread in a CTA.

Semantics

The compiler guarantees that this limit will not be exceeded. The actual number of registers used
may be less; for example, the backend may be able to compile to fewer registers, or the maximum
number of registers may be further constrained by `.maxntid` and `.maxctapersm`.

PTX ISA Notes

Introduced in PTX ISA version 1.3.

Target ISA Notes

Supported on all target architectures.

Examples

```
.entry foo .maxnreg 16 { ... }  // max regs per thread = 16
```

Copy to clipboard
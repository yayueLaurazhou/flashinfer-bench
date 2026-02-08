# 9.7.19.1. Miscellaneous Instructions: brkpt

#### 9.7.19.1. [Miscellaneous Instructions: `brkpt`](https://docs.nvidia.com/cuda/parallel-thread-execution/#miscellaneous-instructions-brkpt)[](https://docs.nvidia.com/cuda/parallel-thread-execution/#miscellaneous-instructions-brkpt "Permalink to this headline")

`brkpt`

Breakpoint.

Syntax

```
brkpt;
```

Copy to clipboard

Description

Suspends execution.

PTX ISA Notes

Introduced in PTX ISA version 1.0.

Target ISA Notes

`brkpt` requires `sm_11` or higher.

Examples

```
    brkpt;
@p  brkpt;
```

Copy to clipboard
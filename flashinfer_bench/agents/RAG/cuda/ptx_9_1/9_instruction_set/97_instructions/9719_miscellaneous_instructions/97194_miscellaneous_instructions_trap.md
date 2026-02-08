# 9.7.19.4. Miscellaneous Instructions: trap

#### 9.7.19.4. [Miscellaneous Instructions: `trap`](https://docs.nvidia.com/cuda/parallel-thread-execution/#miscellaneous-instructions-trap)[](https://docs.nvidia.com/cuda/parallel-thread-execution/#miscellaneous-instructions-trap "Permalink to this headline")

`trap`

Perform trap operation.

Syntax

```
trap;
```

Copy to clipboard

Description

Abort execution and generate an interrupt to the host CPU.

PTX ISA Notes

Introduced in PTX ISA version 1.0.

Target ISA Notes

Supported on all target architectures.

Examples

```
    trap;
@p  trap;
```

Copy to clipboard
# 9.7.12.7. Control Flow Instructions: exit

#### 9.7.12.7. [Control Flow Instructions: `exit`](https://docs.nvidia.com/cuda/parallel-thread-execution/#control-flow-instructions-exit)[](https://docs.nvidia.com/cuda/parallel-thread-execution/#control-flow-instructions-exit "Permalink to this headline")

`exit`

Terminate a thread.

Syntax

```
exit;
```

Copy to clipboard

Description

Ends execution of a thread.

Barriers exclusively waiting on arrivals from exited threads are always released.

PTX ISA Notes

Introduced in PTX ISA version 1.0.

Target ISA Notes

Supported on all target architectures.

Examples

```
    exit;
@p  exit;
```

Copy to clipboard
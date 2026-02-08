# 9.7.13.11. Parallel Synchronization and Communication Instructions: activemask

#### 9.7.13.11. [Parallel Synchronization and Communication Instructions: `activemask`](https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-activemask)[](https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-activemask "Permalink to this headline")

`activemask`

Queries the active threads within a warp.

Syntax

```
activemask.b32 d;
```

Copy to clipboard

Description

`activemask` queries predicated-on active threads from the executing warp and sets the destination
`d` with 32-bit integer mask where bit position in the mask corresponds to the thread’s
`laneid`.

Destination `d` is a 32-bit destination register.

An active thread will contribute 1 for its entry in the result and exited or inactive or
predicated-off thread will contribute 0 for its entry in the result.

PTX ISA Notes

Introduced in PTX ISA version 6.2.

Target ISA Notes

Requires `sm_30` or higher.

Examples

```
activemask.b32  %r1;
```

Copy to clipboard
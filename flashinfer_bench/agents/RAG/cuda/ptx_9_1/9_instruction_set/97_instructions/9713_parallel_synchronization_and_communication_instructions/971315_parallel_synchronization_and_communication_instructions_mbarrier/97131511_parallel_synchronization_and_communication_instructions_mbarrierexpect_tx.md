# 9.7.13.15.11. Parallel Synchronization and Communication Instructions: mbarrier.expect_tx

##### 9.7.13.15.11. [Parallel Synchronization and Communication Instructions: `mbarrier.expect_tx`](https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-mbarrier-expect-tx)[](https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-mbarrier-expect-tx "Permalink to this headline")

`mbarrier.expect_tx`

Perfoms
[expect-tx](https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-mbarrier-expect-tx-operation)
operation on the *mbarrier object*.

Syntax

```
mbarrier.expect_tx{.sem.scope}{.space}.b64 [addr], txCount;

.sem   = { .relaxed }
.scope = { .cta, .cluster }
.space = { .shared{::cta}, .shared::cluster }
```

Copy to clipboard

Description

A thread executing `mbarrier.expect_tx` performs an [expect-tx](https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-mbarrier-expect-tx-operation)
operation on the *mbarrier object* at the location specified by the address operand `addr`. The
32-bit unsigned integer operand `txCount` specifies the `expectCount` argument to the
*expect-tx* operation.

If no state space is specified then [Generic Addressing](https://docs.nvidia.com/cuda/parallel-thread-execution/#generic-addressing) is
used. If the address specified by `addr` does not fall within the address window of
`.shared::cta` or `.shared::cluster` state space then the behavior is undefined.

Supported addressing modes for operand `addr` are as described in [Addresses as Operands](https://docs.nvidia.com/cuda/parallel-thread-execution/#addresses-as-operands).
Alignment for operand `addr` is as described in the
[Size and alignment of mbarrier object](https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-mbarrier-size-alignment).

The optional `.sem` qualifier specifies a memory synchronizing effect as described in the
[Memory Consistency Model](https://docs.nvidia.com/cuda/parallel-thread-execution/#memory-consistency-model).
The `.relaxed` qualifier does not provide any memory ordering semantics and visibility
guarantees.

The optional `.scope` qualifier indicates the set of threads that directly observe the memory
synchronizing effect of this operation, as described in the [Memory Consistency Model](https://docs.nvidia.com/cuda/parallel-thread-execution/#memory-consistency-model).

Qualifiers `.sem` and `.scope` must be specified together.

PTX ISA Notes

Introduced in PTX ISA version 8.0.

Target ISA Notes

Requires `sm_90` or higher.

Examples

```
mbarrier.expect_tx.b64                       [addr], 32;
mbarrier.expect_tx.relaxed.cta.shared.b64    [mbarObj1], 512;
mbarrier.expect_tx.relaxed.cta.shared.b64    [mbarObj2], 512;
```

Copy to clipboard
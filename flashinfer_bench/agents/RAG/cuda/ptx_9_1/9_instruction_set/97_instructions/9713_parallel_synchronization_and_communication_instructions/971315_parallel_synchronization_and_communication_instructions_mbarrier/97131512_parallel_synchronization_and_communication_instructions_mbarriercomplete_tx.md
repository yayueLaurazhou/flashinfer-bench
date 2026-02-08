# 9.7.13.15.12. Parallel Synchronization and Communication Instructions: mbarrier.complete_tx

##### 9.7.13.15.12. [Parallel Synchronization and Communication Instructions: `mbarrier.complete_tx`](https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-mbarrier-complete-tx)[](https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-mbarrier-complete-tx "Permalink to this headline")

`mbarrier.complete_tx`

Perfoms
[complete-tx](https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-mbarrier-complete-tx-operation)
operation on the *mbarrier object*.

Syntax

```
mbarrier.complete_tx{.sem.scope}{.space}.b64 [addr], txCount;

.sem   = { .relaxed }
.scope = { .cta, .cluster }
.space = { .shared{::cta}, .shared::cluster }
```

Copy to clipboard

Description

A thread executing `mbarrier.complete_tx` performs a [complete-tx](https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-mbarrier-complete-tx-operation)
operation on the *mbarrier object* at the location specified by the address operand `addr`. The
32-bit unsigned integer operand `txCount` specifies the `completeCount` argument to the
*complete-tx* operation.

`mbarrier.complete_tx` does not involve any asynchronous memory operations and only simulates the
completion of an asynchronous memory operation and its side effect of signaling to the *mbarrier
object*.

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
mbarrier.complete_tx.b64             [addr],     32;
mbarrier.complete_tx.shared.b64      [mbarObj1], 512;
mbarrier.complete_tx.relaxed.cta.b64 [addr2],    32;
```

Copy to clipboard
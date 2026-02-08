# 9.7.9.27. Data Movement and Conversion Instructions: multimem.cp.reduce.async.bulk

#### 9.7.9.27. [Data Movement and Conversion Instructions: `multimem.cp.reduce.async.bulk`](https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-multimem-cp-reduce-async-bulk)[](https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-multimem-cp-reduce-async-bulk "Permalink to this headline")

`multimem.cp.reduce.async.bulk`

Initiates an asynchronous reduction operation to a multimem address range.

Syntax

```
multimem.cp.reduce.async.bulk.dst.src.completion_mechanism.redOp.type  [dstMem], [srcMem], size;

    .dst                  = { .global }
    .src                  = { .shared::cta }
    .completion_mechanism = { .bulk_group }
    .redOp                = { .and, .or, .xor,
                              .add, .inc, .dec,
                              .min, .max }
    .type                 = { .f16, .bf16,
                              .b32, .u32, .s32,
                              .b64, .u64, .s64,
                              .f32, .f64 }

multimem.cp.reduce.async.bulk.dst.src.completion_mechanism.add.noftz.type  [dstMem], [srcMem], size;

    .dst                  = { .global }
    .src                  = { .shared::cta }
    .completion_mechanism = { .bulk_group }
    .type                 = { .f16, .bf16 }
```

Copy to clipboard

Description

Instruction `multimem.cp.reduce.async.bulk` initiates an element-wise asynchronous reduction
operation with elements from source memory address range `[srcMem, srcMem + size)` to memory
locations residing on each GPU’s memory referred to by the multimem destination address range
`[dstMem, dstMem + size)`.

Each data element in the destination array is reduced inline with the corresponding data element in
the source array with the reduction operation specified by the modifier `.redOp`. The type of each
data element in the source and the destination array is specified by the modifier `.type`.

The source address operand `srcMem` is in the state space specified by `.src` and the
destination address operand `dstMem` is in the state specified by the `.dst`.

The 32-bit operand `size` specifies the amount of memory to be copied from the source location
and used in the reduction operation, in terms of number of bytes. Operand `size` must be a
multiple of 16. The memory range `[dstMem, dstMem + size)` must not overflow the destination
multimem memory space. The memory range `[srcMem, srcMem + size)` must not overflow the source
memory space. The addresses `dstMem` and `srcMem` must be aligned to 16 bytes. If any of these
preconditions is not met, the behavior is undefined.

The operations supported by `.redOp` are classified as follows:

The bit-size operations are `.and`, `.or`, and `.xor`.

The integer operations are `.add`, `.inc`, `.dec`, `.min`, and `.max`. The `.inc` and
`.dec` operations return a result in the range `[0..x]` where `x` is the value at the source
state space.

The floating point operation `.add` rounds to the nearest even, preserve input and result
subnormals, and does not flush them to zero, except for the current implementation of
`multimem.cp.reduce.async.bulk.add.f32` which flushes subnormal inputs and results to
sign-preserving zero. The `multimem.cp.reduce.async.bulk.add.f16` and
`multimem.cp.reduce.async.bulk.add.bf16` operations require `.noftz` qualifier. It preserves
input and result subnormals, and does not flush them to zero.

The following table describes the valid combinations of `.redOp` and element type:

| .redOp | element type |
| --- | --- |
| .add | .u32, .s32, .u64, .f32, .f64, .f16, .bf16 |
| .min, .max | .u32, .s32, .u64, .s64, .f16, .bf16 |
| .inc, .dec | .u32 |
| .and, .or, .xor | .b32, .b64 |

The modifier `.completion_mechanism` specifies the completion mechanism that is supported by the
instruction. The modifier `.bulk_group` specifies that the `multimem.cp.reduce.async.bulk` uses
bulk async-group based completion mechanism.

Each reduction operation performed by the `multimem.cp.reduce.async.bulk` has individually
`.relaxed.sys` memory ordering semantics. The load operations in `multimem.cp.reduce.async.bulk`
are treated as weak memory operations as described in the [Memory Consistency Model](https://docs.nvidia.com/cuda/parallel-thread-execution/#memory-consistency-model).

PTX ISA Notes

Introduced in PTX ISA version 9.1.

Target ISA Notes

Requires `sm_90` or higher.

Examples

```
multimem.cp.reduce.async.bulk.global.shared::cta.bulk_group.add.u32 [dstMem], [srcMem], size;

multimem.cp.reduce.async.bulk.global.shared::cta.bulk_group.xor.b64 [dstMem], [srcMem], size;

multimem.cp.reduce.async.bulk.global.shared::cta.bulk_group.inc.u32 [dstMem], [srcMem], size;

multimem.cp.reduce.async.bulk.global.shared::cta.bulk_group.dec.u32 [dstMem], [srcMem], size;

multimem.cp.reduce.async.bulk.global.shared::cta.bulk_group.max.s32 [dstMem], [srcMem], size;

multimem.cp.reduce.async.bulk.global.shared::cta.bulk_group.add.noftz.f16 [dstMem], [srcMem], size;

multimem.cp.reduce.async.bulk.global.shared::cta.bulk_group.min.bf16 [dstMem], [srcMem], size;

multimem.cp.reduce.async.bulk.global.shared::cta.bulk_group.add.noftz.bf16 [dstMem], [srcMem], size;
```

Copy to clipboard
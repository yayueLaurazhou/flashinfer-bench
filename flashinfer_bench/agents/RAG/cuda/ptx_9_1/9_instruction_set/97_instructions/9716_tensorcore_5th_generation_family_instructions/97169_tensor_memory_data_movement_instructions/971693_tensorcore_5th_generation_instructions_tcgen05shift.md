# 9.7.16.9.3. Tensorcore 5th Generation Instructions: tcgen05.shift

##### 9.7.16.9.3. [Tensorcore 5th Generation Instructions: `tcgen05.shift`](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-instructions-tcgen05-shift)[](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-instructions-tcgen05-shift "Permalink to this headline")

`tcgen05.shift`

Asynchronously shift down the rows of the matrix in the [Tensor Memory](https://docs.nvidia.com/cuda/parallel-thread-execution/#tensor-memory) for a warp.

Syntax

```
tcgen05.shift.cta_group.down  [taddr];

.cta_group = { .cta_group::1, .cta_group::2 }
```

Copy to clipboard

Description

Instruction `tcgen05.shift` is an asynchronous instruction which initiates the shifting of 32-byte
elements downwards across all the rows, except the last, by one row. The address operand `taddr`
specifies the base address of the matrix in the [Tensor Memory](https://docs.nvidia.com/cuda/parallel-thread-execution/#tensor-memory) whose rows must
be down shifted.

The lane of the address operand `taddr` must be aligned to 32.

Qualifier `.cta_group` specifies the number of CTAs whose [Tensor Memory](https://docs.nvidia.com/cuda/parallel-thread-execution/#tensor-memory)
is touched when a single thread of a single CTA executes the `tcgen05.shift` instruction.
When `.cta_group::1` is specified, the shift operation is performed in the
[Tensor Memory](https://docs.nvidia.com/cuda/parallel-thread-execution/#tensor-memory) of the current CTA. When `.cta_group::2` is specified,
the shift operation is performed in the [Tensor Memory](https://docs.nvidia.com/cuda/parallel-thread-execution/#tensor-memory) of both the current and the
[peer CTAs](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-peer-cta).

All `tcgen05` instructions within a kernel must specify the same value for the `.cta_group`
qualifier.

PTX ISA Notes

Introduced in PTX ISA version 8.6.

Target ISA Notes

Supported on following architectures:

* `sm_100a`
* `sm_101a` (Renamed to `sm_110a` from PTX ISA version 9.0)
* `sm_103a`
* `sm_110a`

Examples

```
tcgen05.shift.down.cta_group::1 [taddr0];
tcgen05.shift.down.cta_group::2 [taddr1];
```

Copy to clipboard
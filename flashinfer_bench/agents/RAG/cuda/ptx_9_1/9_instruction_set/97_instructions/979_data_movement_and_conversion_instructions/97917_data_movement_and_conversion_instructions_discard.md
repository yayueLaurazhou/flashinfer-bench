# 9.7.9.17. Data Movement and Conversion Instructions: discard

#### 9.7.9.17. [Data Movement and Conversion Instructions: `discard`](https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-discard)[](https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-discard "Permalink to this headline")

`discard`

Discard the data at the specified address range and cache level.

Syntax

```
discard{.global}.level  [a], size;

.level = { .L2 };
```

Copy to clipboard

Description

Semantically, this behaves like a weak write of an *unstable indeterminate value*:
reads of memory locations with *unstable indeterminate values* may return different
bit patterns each time until the memory is overwritten.
This operation *hints* to the implementation that data in the specified cache `.level`
can be destructively discarded without writing it back to memory.

The operand `size` is an integer constant that specifies the length in bytes of the
address range `[a, a + size)` to write *unstable indeterminate values* into.
The only supported value for the `size` operand is `128`.

If no state space is specified then [Generic Addressing](https://docs.nvidia.com/cuda/parallel-thread-execution/#generic-addressing) is used.
If the specified address does not fall within the address window of `.global` state space
then the behavior is undefined.

Supported addressing modes for address operand `a` are described in [Addresses as Operands](https://docs.nvidia.com/cuda/parallel-thread-execution/#addresses-as-operands).
`a` must be aligned to 128 bytes.

PTX ISA Notes

Introduced in PTX ISA version 7.4.

Target ISA Notes

Requires `sm_80` or higher.

Examples

```
discard.global.L2 [ptr], 128;
ld.weak.u32 r0, [ptr];
ld.weak.u32 r1, [ptr];
// The values in r0 and r1 may differ!
```

Copy to clipboard
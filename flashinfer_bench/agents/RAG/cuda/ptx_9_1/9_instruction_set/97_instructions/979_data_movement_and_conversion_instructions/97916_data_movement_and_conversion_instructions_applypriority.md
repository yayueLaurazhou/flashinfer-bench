# 9.7.9.16. Data Movement and Conversion Instructions: applypriority

#### 9.7.9.16. [Data Movement and Conversion Instructions: `applypriority`](https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-applypriority)[](https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-applypriority "Permalink to this headline")

`applypriority`

Apply the cache eviction priority to the specified address in the specified cache level.

Syntax

```
applypriority{.global}.level::eviction_priority  [a], size;

.level::eviction_priority = { .L2::evict_normal };
```

Copy to clipboard

Description

The `applypriority` instruction applies the cache eviction priority specified by the
`.level::eviction_priority` qualifier to the address range `[a..a+size)` in the specified cache
level.

If no state space is specified then [Generic Addressing](https://docs.nvidia.com/cuda/parallel-thread-execution/#generic-addressing) is
used. If the specified address does not fall within the address window of `.global` state space
then the behavior is undefined.

The operand `size` is an integer constant that specifies the amount of data, in bytes, in the
specified cache level on which the priority is to be applied. The only supported value for the
`size` operand is 128.

Supported addressing modes for operand `a` are described in [Addresses as Operands](https://docs.nvidia.com/cuda/parallel-thread-execution/#addresses-as-operands).
`a` must be aligned to 128 bytes.

PTX ISA Notes

Introduced in PTX ISA version 7.4.

Target ISA Notes

Requires `sm_80` or higher.

Examples

```
applypriority.global.L2::evict_normal [ptr], 128;
```

Copy to clipboard
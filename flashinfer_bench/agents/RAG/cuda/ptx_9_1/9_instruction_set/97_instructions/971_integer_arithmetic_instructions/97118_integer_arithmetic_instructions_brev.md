# 9.7.1.18. Integer Arithmetic Instructions: brev

#### 9.7.1.18. [Integer Arithmetic Instructions: `brev`](https://docs.nvidia.com/cuda/parallel-thread-execution/#integer-arithmetic-instructions-brev)[](https://docs.nvidia.com/cuda/parallel-thread-execution/#integer-arithmetic-instructions-brev "Permalink to this headline")

`brev`

Bit reverse.

Syntax

```
brev.type  d, a;

.type = { .b32, .b64 };
```

Copy to clipboard

Description

Perform bitwise reversal of input.

Semantics

```
msb = (.type==.b32) ? 31 : 63;

for (i=0; i<=msb; i++) {
    d[i] = a[msb-i];
}
```

Copy to clipboard

PTX ISA Notes

Introduced in PTX ISA version 2.0.

Target ISA Notes

`brev` requires `sm_20` or higher.

Examples

```
brev.b32  d, a;
```

Copy to clipboard
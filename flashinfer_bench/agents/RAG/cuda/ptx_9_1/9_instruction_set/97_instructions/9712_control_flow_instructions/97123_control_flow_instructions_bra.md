# 9.7.12.3. Control Flow Instructions: bra

#### 9.7.12.3. [Control Flow Instructions: `bra`](https://docs.nvidia.com/cuda/parallel-thread-execution/#control-flow-instructions-bra)[](https://docs.nvidia.com/cuda/parallel-thread-execution/#control-flow-instructions-bra "Permalink to this headline")

`bra`

Branch to a target and continue execution there.

Syntax

```
@p   bra{.uni}  tgt;           // tgt is a label
     bra{.uni}  tgt;           // unconditional branch
```

Copy to clipboard

Description

Continue execution at the target. Conditional branches are specified by using a guard predicate. The
branch target must be a label.

`bra.uni` is guaranteed to be non-divergent, i.e. all active threads in a warp that are currently
executing this instruction have identical values for the guard predicate and branch target.

Semantics

```
if (p) {
    pc = tgt;
}
```

Copy to clipboard

PTX ISA Notes

Introduced in PTX ISA version 1.0.

Unimplemented indirect branch introduced in PTX ISA version 2.1 has been removed from the spec.

Target ISA Notes

Supported on all target architectures.

Examples

```
bra.uni  L_exit;    // uniform unconditional jump
@q  bra      L23;   // conditional branch
```

Copy to clipboard
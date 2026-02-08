# 11.4.7. Performance-Tuning Directives: .pragma

### 11.4.7. [Performance-Tuning Directives: `.pragma`](https://docs.nvidia.com/cuda/parallel-thread-execution/#performance-tuning-directives-pragma)[](https://docs.nvidia.com/cuda/parallel-thread-execution/#performance-tuning-directives-pragma "Permalink to this headline")

`.pragma`

Pass directives to PTX backend compiler.

Syntax

```
.pragma list-of-strings ;
```

Copy to clipboard

Description

Pass module-scoped, entry-scoped, or statement-level directives to the PTX backend compiler.

The `.pragma` directive may occur at module-scope, at entry-scope, or at statement-level.

Semantics

The interpretation of `.pragma` directive strings is implementation-specific and has no impact on
PTX semantics. See [Descriptions of .pragma Strings](https://docs.nvidia.com/cuda/parallel-thread-execution/#descriptions-pragma-strings) for
descriptions of the pragma strings defined in `ptxas`.

PTX ISA Notes

Introduced in PTX ISA version 2.0.

Target ISA Notes

Supported on all target architectures.

Examples

```
.pragma "nounroll";    // disable unrolling in backend

// disable unrolling for current kernel
.entry foo .pragma "nounroll"; { ... }
```

Copy to clipboard